"""
UploadM8 Thumbnail Stage — AI-Powered Universal Edition
========================================================
Generate multiple candidate thumbnails, score each frame for sharpness,
then run AI-powered content-category-aware selection to pick the most
algorithmically compelling frame — not just the sharpest one.

Flow:
  1. Probe video duration via ffprobe
  2. Distribute N frame offsets across the video (N = tier max_thumbnails)
  3. Extract each frame as a 1080px-wide JPEG
  4. Score each frame with FFmpeg blurdetect (higher = sharper)
  5. Detect content category (3-layer: user hint → filename → general)
  6. AI selection pass — send all candidates to GPT-4o-mini with
     category-specific selection criteria (picks the most ENGAGING frame
     for the content type, not just the sharpest)
  7. Set ctx.thumbnail_path  = AI-selected (or sharpest as fallback)
     Set ctx.thumbnail_paths = all candidates in chronological order
     Set ctx.thumbnail_scores = {str(path): score} for all candidates
  8. Store metadata in ctx.output_artifacts for queue UI display
  9. [When can_custom_thumbnails] Generate Thumbnail Brief (JSON) via GPT,
     render MrBeast-style composite (headline, badge, arrow) per platform:
     YouTube 16:9, Instagram/Facebook 9:16 center-safe, TikTok thumb_offset only.
     Render via upgraded PIL template (gradient + type) or AI image edit (when can_ai_thumbnail_styling).

AI Selection Criteria (per category):
  beauty      — best lighting, eyes open, makeup clearly visible
  food        — most appetizing shot, food filling frame, vivid color
  gaming      — peak action, dramatic moment, HUD/score visible
  automotive  — peak speed feel, road visible, most dynamic angle
  fitness     — peak effort, form clearly visible, sweat/exertion present
  travel      — widest most scenic, landmark clearly identifiable
  fashion     — full outfit visible, best pose, clean background
  comedy      — peak reaction expression, comedic climax moment
  pets        — eyes visible, peak cuteness/action, animal in focus
  education   — presenter engaged, key graphic/diagram readable
  general     — highest visual impact, most representative of content

Fallback chain:
  - AI unavailable (no API key or plan gate): use sharpest frame (blurdetect)
  - blurdetect fails: use file-size proxy (larger = more detail)
  - Extraction fails at all offsets: retry at t=0
  - Everything fails: raise SkipStage (non-fatal — pipeline continues)

NOTE: R2 upload and DB persistence are handled by worker.py AFTER this stage.
This stage only writes to ctx — it never touches the database or R2 directly.

Exports: run_thumbnail_stage(ctx)
"""

import asyncio
import base64
import copy
import hashlib
import html as html_module
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import context as _context_mod
from .context import JobContext, resolve_fused_thumbnail_category
from .entitlements import should_generate_thumbnails
from .errors import SkipStage
from services.ml_strategy_utils import prefer_ai_thumbnail_vs_sharpness
from .pikzels_api import studio_renderer_enabled, render_thumbnail_with_studio_renderer
from .trend_intel import fetch_trend_intel
from .thumbnail_qa import (
    YOUTUBE_SEARCH_PREVIEW_QA,
    assess_youtube_search_preview_readability,
    pick_tiktok_cover_offset_seconds,
)
from .safe_parse import json_dict

logger = logging.getLogger("uploadm8-worker.thumbnail")

# Import-compat guard:
# some environments may still export the typo alias (THUMNAIL_BRIEF_PROMPT)
# while others export the corrected constant (THUMBNAIL_BRIEF_PROMPT).
THUMBNAIL_BRIEF_PROMPT = getattr(
    _context_mod,
    "THUMBNAIL_BRIEF_PROMPT",
    getattr(_context_mod, "THUMNAIL_BRIEF_PROMPT", ""),
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_THUMBNAIL_OFFSET = 1.0
MAX_THUMBNAIL_OFFSET     = 300.0
MIN_THUMB_SIZE           = 2048          # bytes — smaller = rejected
OPENAI_API_KEY           = os.environ.get("OPENAI_API_KEY", "")
OPENAI_THUMB_MODEL       = os.environ.get("OPENAI_THUMB_MODEL", "gpt-4o-mini")
# Images /edits API now requires an explicit model (e.g. gpt-image-1, dall-e-2).
OPENAI_IMAGE_EDIT_MODEL  = os.environ.get("OPENAI_IMAGE_EDIT_MODEL", "gpt-image-1")
# Optional bold font for template thumbnails (falls back to Arial / DejaVu).
THUMBNAIL_FONT_BOLD = os.environ.get("THUMBNAIL_FONT_BOLD", "").strip()


def _thumbnail_render_engine_mode() -> str:
    """
    Select styled-thumbnail renderer:
    - internal: use UploadM8 native renderer stack
    - studio: try external studio renderer first, then fallback to internal
    - auto: same as studio when configured; otherwise internal
    """
    raw = str(os.environ.get("THUMB_RENDER_ENGINE", "studio")).strip().lower()
    # Legacy engine token support.
    if raw == "pikzels":
        return "studio"
    if raw in ("internal", "studio", "auto"):
        return raw
    return "internal"


# ============================================================
# Content Category Engine
# ============================================================
# Defined independently here so thumbnail_stage has zero circular imports.
# Mirrors the same detection system in caption_stage.py.

_THUMB_CATEGORIES: Dict[str, Dict] = {
    "automotive": {
        "keywords": [
            "car", "drive", "driving", "road", "highway", "speed", "mph",
            "truck", "suv", "motorcycle", "moto", "drift", "race", "track",
            "vehicle", "auto", "engine", "trill", "dashcam", "cruise",
            "roadtrip", "throttle", "turbo", "supercar", "offroad", "joyride",
        ],
        "selection_criteria": (
            "Select the frame that BEST conveys speed and excitement: "
            "peak dynamic moment on a road, vehicle clearly in motion, "
            "speedometer or HUD prominent if present, dramatic angle. "
            "Avoid static parked shots or boring straight-road frames."
        ),
    },
    "beauty": {
        "keywords": [
            "makeup", "beauty", "skincare", "foundation", "concealer", "blush",
            "lipstick", "eyeshadow", "mascara", "eyeliner", "contour", "glam",
            "grwm", "glow", "bronzer", "primer", "serum", "toner",
        ],
        "selection_criteria": (
            "Select the frame with the BEST LIGHTING on the subject's face: "
            "eyes wide open and expressive, makeup clearly and fully applied, "
            "confident or radiant look direct to camera. "
            "Avoid mid-blink, blurry transition, or raw skin 'before' frames. "
            "The final 'after' reveal frame almost always wins."
        ),
    },
    "food": {
        "keywords": [
            "food", "recipe", "cook", "cooking", "bake", "baking", "eat",
            "meal", "dinner", "lunch", "breakfast", "snack", "restaurant",
            "foodie", "chef", "kitchen", "dish", "dessert", "cake", "pasta",
            "steak", "pizza", "burger", "sushi",
        ],
        "selection_criteria": (
            "Select the frame that makes the food look most APPETIZING: "
            "finished dish filling the frame, vivid colors, steam or glossy texture visible, "
            "great plating and lighting. "
            "Avoid raw ingredients, messy cutting boards, or empty pans. "
            "The finished hero shot is almost always the best choice."
        ),
    },
    "home_renovation": {
        "keywords": [
            "reno", "renovation", "diy", "makeover", "before after", "transform",
            "decor", "interior", "design", "build", "tile", "paint", "floor",
            "wall", "cabinet", "remodel", "woodwork", "carpentry",
        ],
        "selection_criteria": (
            "Select the frame showing the most DRAMATIC TRANSFORMATION: "
            "the finished, clean, complete room or space at its best angle. "
            "Avoid mid-construction debris, drop cloths, or unfinished surfaces. "
            "Wide establishing shot of the completed space wins almost every time."
        ),
    },
    "gardening": {
        "keywords": [
            "garden", "gardening", "plant", "plants", "grow", "flower",
            "flowers", "vegetable", "herb", "seed", "soil", "harvest",
            "greenhouse", "bloom", "prune", "mulch",
        ],
        "selection_criteria": (
            "Select the frame with the most LUSH or VIBRANT plant life: "
            "in-bloom flowers, ripe harvest, full green growth filling the frame. "
            "Avoid bare soil, empty seedling trays, or sparse beds. "
            "Best color saturation and fullest frame of living plants wins."
        ),
    },
    "fitness": {
        "keywords": [
            "workout", "gym", "fitness", "exercise", "train", "lift", "lifting",
            "weights", "cardio", "run", "yoga", "pilates", "hiit", "crossfit",
            "strength", "gains", "physique", "sweat", "reps",
        ],
        "selection_criteria": (
            "Select the frame showing PEAK EFFORT: maximum exertion clearly visible, "
            "form correct and muscles engaged, sweat present, intense expression. "
            "High-energy action beats a resting frame. "
            "Avoid warmup stretches, rest periods, or talking-head setup segments."
        ),
    },
    "fashion": {
        "keywords": [
            "fashion", "outfit", "ootd", "style", "clothes", "haul", "thrift",
            "fitcheck", "lookbook", "trend", "streetwear", "vintage", "aesthetic",
        ],
        "selection_criteria": (
            "Select the frame where the FULL OUTFIT is most visible: "
            "head-to-toe or torso-down clearly framed, best pose, "
            "clean or visually interesting background. "
            "Avoid partial crops, mid-change transitions, or obscured clothing. "
            "Confident pose with outfit fully featured wins."
        ),
    },
    "gaming": {
        "keywords": [
            "game", "gaming", "gamer", "gameplay", "stream", "fps", "rpg",
            "controller", "xbox", "playstation", "nintendo", "fortnite",
            "minecraft", "valorant", "cod", "lol", "roblox", "speedrun", "esports",
        ],
        "selection_criteria": (
            "Select the frame with the highest VISUAL DRAMA: "
            "peak action, critical game moment, impressive score or stat on screen, "
            "dramatic HUD state, or the most exciting environment visible. "
            "Avoid menu screens, loading screens, or quiet exploration frames. "
            "The moment of impact, victory, or highest tension wins."
        ),
    },
    "travel": {
        "keywords": [
            "travel", "trip", "vacation", "destination", "explore", "adventure",
            "abroad", "beach", "mountain", "hotel", "backpack", "sightseeing",
            "landmark", "scenic", "wanderlust", "passport",
        ],
        "selection_criteria": (
            "Select the frame with the most STUNNING SCENERY or ICONIC LOCATION: "
            "widest angle capturing the full landscape, most recognizable landmark, "
            "best natural lighting — golden hour or dramatic sky preferred. "
            "Avoid cramped airport shots, hotel rooms, or ordinary streets. "
            "The 'money shot' of the destination wins."
        ),
    },
    "pets": {
        "keywords": [
            "dog", "cat", "pet", "puppy", "kitten", "animal", "paw", "bark",
            "meow", "bird", "hamster", "bunny", "rabbit", "furry", "doggo",
            "furryfriend",
        ],
        "selection_criteria": (
            "Select the frame where the pet's EYES AND FACE are most clearly visible: "
            "cute or funny expression, mid-action (jump, play, zoomies), or an emotional moment. "
            "Avoid frames where the animal is turned away, far in background, or partially hidden. "
            "Direct eye contact with the camera is almost always the single best thumbnail."
        ),
    },
    "education": {
        "keywords": [
            "learn", "learning", "teach", "tutorial", "how to", "tips", "tricks",
            "hacks", "guide", "course", "study", "skill", "knowledge", "fact",
            "science", "history", "psychology", "finance", "productivity",
        ],
        "selection_criteria": (
            "Select the frame where the presenter is MOST ENGAGED and CLEARLY VISIBLE: "
            "expressive face showing the key insight, key graphic or diagram readable behind them, "
            "or the single most important visual in the tutorial. "
            "Avoid generic blank-background talking-head frames. "
            "The 'aha moment' expression or the key visual element wins."
        ),
    },
    "comedy": {
        "keywords": [
            "funny", "comedy", "joke", "prank", "skit", "reaction", "meme",
            "laugh", "hilarious", "parody", "relatable", "pov",
        ],
        "selection_criteria": (
            "Select the frame with the PEAK REACTION or highest-energy expression: "
            "biggest laugh, most exaggerated face, or the comedic climax moment. "
            "Over-the-top expressions and reactions drive the most clicks. "
            "Avoid setup frames, neutral expressions, or flat delivery moments."
        ),
    },
    "tech": {
        "keywords": [
            "tech", "technology", "app", "software", "phone", "laptop", "computer",
            "review", "unboxing", "setup", "desk setup", "ai", "gadget", "gear",
        ],
        "selection_criteria": (
            "Select the frame that best SHOWCASES THE PRODUCT or KEY VISUAL: "
            "product in sharp focus and well-lit, screen content clearly visible, "
            "the 'wow' feature or unboxing reveal moment front and center. "
            "Avoid blurry close-ups, hands obscuring the product, "
            "or generic presenter-only shots with no product visible."
        ),
    },
    "music": {
        "keywords": [
            "music", "song", "singing", "sing", "cover", "original", "produce",
            "beat", "studio", "record", "guitar", "piano", "drums", "vocal",
            "concert", "gig", "performance",
        ],
        "selection_criteria": (
            "Select the frame at PEAK MUSICAL INTENSITY: "
            "emotional climax, instrument mid-play at its most dynamic, "
            "vocalist at their most expressive, or the most cinematic "
            "studio or performance shot. "
            "Avoid setup, instrument-tuning, or low-energy talking frames. "
            "Performance emotion and energy drive clicks on music content."
        ),
    },
    "real_estate": {
        "keywords": [
            "real estate", "property", "house", "home", "apartment", "listing",
            "for sale", "rent", "mortgage", "investing", "flip", "flipping",
            "rental", "cashflow",
        ],
        "selection_criteria": (
            "Select the frame showing the property's BEST FEATURE or MONEY SHOT: "
            "most impressive room (kitchen, pool, master suite, view), "
            "best curb appeal exterior, or the most aspirational space. "
            "Avoid cluttered rooms, dark spaces, or talking-head-only frames. "
            "The frame that makes viewers want to tour the property wins."
        ),
    },
    "sports": {
        "keywords": [
            "sport", "sports", "athlete", "soccer", "football", "basketball",
            "baseball", "tennis", "golf", "swim", "hockey", "mma", "boxing",
            "tournament", "league", "training", "score", "goal",
        ],
        "selection_criteria": (
            "Select the frame at the PEAK ATHLETIC MOMENT: "
            "ball in air, mid-strike, maximum effort face, celebration, "
            "or the single most dramatic instant of the entire play sequence. "
            "Avoid warmup, sideline conversations, or static standing shots. "
            "The apex of action is the thumbnail that gets the click."
        ),
    },
    "asmr": {
        "keywords": [
            "asmr", "satisfying", "relaxing", "calm", "soothing", "triggers",
            "tingles", "whisper", "tapping", "crunchy", "slime", "soap",
            "oddly satisfying",
        ],
        "selection_criteria": (
            "Select the frame that looks most VISUALLY SATISFYING or TEXTURAL: "
            "peak satisfying moment (soap cut, slime stretch, tapping close-up), "
            "the closest and most interesting texture shot, "
            "or the most calm and aesthetically pleasing composition. "
            "Avoid blurry setups or frames where the primary subject is out of frame."
        ),
    },
    "lifestyle": {
        "keywords": [
            "vlog", "day in my life", "daily", "morning routine", "night routine",
            "productive", "productivity", "life update", "minimalist", "aesthetic",
            "wellness", "self care", "journal",
        ],
        "selection_criteria": (
            "Select the frame that best captures the VIDEO'S VIBE AND AESTHETIC: "
            "presenter looking their best and most genuine, "
            "most aspirational or relatable setting, "
            "clearest representation of what makes this day interesting. "
            "Warm lighting and authentic expression beats posed or studio-looking shots."
        ),
    },
    "general": {
        "keywords": [],   # catch-all — always matches last
        "selection_criteria": (
            "First identify what this video is actually about by examining all the frames carefully. "
            "Then select the frame with the highest VISUAL IMPACT and CLICK-WORTHINESS: "
            "most expressive face or emotion, most dramatic or surprising action, "
            "most visually interesting and colorful composition. "
            "Avoid black frames, mid-transition blurs, low-energy setup shots, "
            "and frames where the subject is not clearly visible."
        ),
    },
}

_CATEGORY_PRIORITY = [
    "automotive", "beauty", "food", "home_renovation", "gardening",
    "fitness", "fashion", "gaming", "travel", "pets", "education",
    "comedy", "tech", "music", "real_estate", "sports", "asmr", "lifestyle",
    "general",
]


def _detect_category(ctx: JobContext) -> str:
    """
    Content category for thumbnails: fusion override → audio/vision canonical →
    user text → filename → general.
    """
    def _scan(text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        for cat in _CATEGORY_PRIORITY:
            if cat == "general":
                continue
            for kw in _THUMB_CATEGORIES[cat].get("keywords", []):
                if kw in t:
                    return cat
        return None

    fused = resolve_fused_thumbnail_category(ctx)
    if fused and fused in _THUMB_CATEGORIES:
        logger.debug(f"Thumbnail category from fusion override: {fused}")
        return fused

    canonical = getattr(ctx, "get_canonical_category", lambda: None)()
    if canonical and canonical in _THUMB_CATEGORIES:
        logger.debug(f"Thumbnail category from canonical (audio/vision): {canonical}")
        return canonical

    for text in (ctx.caption, ctx.title):
        result = _scan(text or "")
        if result:
            logger.debug(f"Thumbnail category from user hint: {result}")
            return result

    result = _scan(ctx.filename or "")
    if result:
        logger.debug(f"Thumbnail category from filename: {result}")
        return result

    logger.debug("Thumbnail category: general (GPT will identify from frames)")
    return "general"


# ============================================================
# AI Thumbnail Selector
# ============================================================

async def _ai_select_best_frame(
    candidates: List[Tuple[Path, float]],
    category: str,
    ctx: JobContext,
) -> Optional[Path]:
    """
    Send all candidate frames to GPT-4o-mini and ask it to select the
    most compelling thumbnail for the detected content category.

    Returns the Path of the AI-selected frame, or None on any failure
    (caller falls back to sharpness scoring).

    Cost: ~$0.01-0.03 per call (low-detail vision, max 8 frames, 100 output tokens).
    """
    if not OPENAI_API_KEY or not candidates:
        return None

    # Cap at 8 frames to control cost — candidates are already in chronological order
    frames_to_send = candidates[:8]
    n = len(frames_to_send)

    # Fetch category-specific selection criteria
    criteria = _THUMB_CATEGORIES.get(
        category, _THUMB_CATEGORIES["general"]
    )["selection_criteria"]

    # Build context hints from job metadata
    context_hints = []
    if ctx.title:
        context_hints.append(f"Video title: {ctx.title}")
    if ctx.caption:
        context_hints.append(f"Creator's caption hint: {ctx.caption}")
    if ctx.location_name:
        context_hints.append(f"Filming location: {ctx.location_name}")
    context_str = (
        f"\nADDITIONAL CONTEXT:\n" + "\n".join(context_hints)
        if context_hints else ""
    )

    prompt_text = (
        f"You are a social media thumbnail expert. Select the single BEST thumbnail "
        f"from {n} video frame candidates.\n\n"
        f"CONTENT CATEGORY: {category.upper().replace('_', ' ')}\n\n"
        f"SELECTION CRITERIA FOR THIS CATEGORY:\n{criteria}\n"
        f"{context_str}\n\n"
        f"You are being shown {n} frames from this video in order (Frame 1 = early in video, "
        f"Frame {n} = later in video).\n\n"
        f"TASK: Examine all {n} frames carefully. Select the ONE frame number that would "
        f"perform best as a thumbnail on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels.\n\n"
        f"Consider: click-through rate potential, category-specific quality (see criteria), "
        f"face/subject clarity, visual composition, color vibrancy, absence of blur or black frames.\n\n"
        f"RESPOND WITH ONLY a JSON object in this exact format, nothing else:\n"
        f'{{\"selected_frame\": 1, \"reason\": \"brief reason\"}}\n\n'
        f"selected_frame must be an integer between 1 and {n}."
    )

    content: List[Dict] = [{"type": "text", "text": prompt_text}]

    attached = 0
    for i, (path, _score) in enumerate(frames_to_send, start=1):
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                }
            })
            # Label each frame so the model can reference it by number
            content.append({
                "type": "text",
                "text": f"[Frame {i}]"
            })
            attached += 1
        except (OSError, PermissionError, TypeError, ValueError) as e:
            logger.debug("Could not attach frame %s: %s", path.name, e)

    if attached == 0:
        logger.warning("AI thumbnail selection: no frames could be attached")
        return None

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_THUMB_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 100,
                    "temperature": 0.2,   # low temp = consistent, deterministic
                },
            )

        if response.status_code != 200:
            logger.warning(
                f"AI thumbnail selection HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            return None

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if GPT wraps in code block
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0]
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0]

        parsed = json_dict(answer, default={}, context="ai_thumbnail_selection")
        try:
            selected_idx = int(parsed.get("selected_frame", 0))
        except (TypeError, ValueError):
            selected_idx = 0
        reason = str(parsed.get("reason", ""))

        if 1 <= selected_idx <= len(frames_to_send):
            selected_path = frames_to_send[selected_idx - 1][0]
            logger.info(
                f"AI thumbnail: Frame {selected_idx}/{n} selected "
                f"(category={category}) — {reason}"
            )
            return selected_path

        logger.warning(
            f"AI thumbnail returned out-of-range index {selected_idx} "
            f"(max={len(frames_to_send)}) — falling back to sharpness"
        )
        return None

    except asyncio.CancelledError:
        raise
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, httpx.HTTPError) as e:
        logger.warning("AI thumbnail selection failed (non-fatal, using sharpness): %s", e)
        return None


# ============================================================
# Thumbnail Brief Generator (platform-aware JSON)
# ============================================================

async def _generate_thumbnail_brief(ctx: JobContext, category: str) -> Optional[Dict]:
    """
    Generate a platform-aware Thumbnail Brief (JSON) using GPT.
    Returns parsed brief dict or None on failure.
    """
    if not OPENAI_API_KEY:
        return None
    vars_ = ctx.get_thumbnail_brief_vars(category=category)
    prompt = THUMBNAIL_BRIEF_PROMPT.format(**vars_)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_THUMB_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.5,
                },
            )
        if response.status_code != 200:
            logger.warning(f"Thumbnail brief HTTP {response.status_code}: {response.text[:200]}")
            return None
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0]
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0]
        brief = json.loads(answer)
        # Ensure platform_plan exists with defaults
        brief.setdefault("platform_plan", {})
        brief["platform_plan"].setdefault("youtube", {"enabled": True, "canvas": "16:9"})
        brief["platform_plan"].setdefault("instagram", {"enabled": True, "canvas": "9:16", "safe_center_pct": 60})
        brief["platform_plan"].setdefault("facebook", {"enabled": True, "canvas": "9:16", "safe_center_pct": 60})
        brief["platform_plan"].setdefault("tiktok", {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5})
        return brief
    except asyncio.CancelledError:
        raise
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, httpx.HTTPError) as e:
        logger.warning("Thumbnail brief generation failed: %s", e)
        return None


# ============================================================
# Template Renderer (PIL overlays — deterministic “last mile” without Photoshop)
# ============================================================

def _thumbnail_font_paths() -> List[str]:
    paths: List[str] = []
    if THUMBNAIL_FONT_BOLD:
        paths.append(THUMBNAIL_FONT_BOLD)
    paths.extend(
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "arial.ttf",
            "Arial.ttf",
        ]
    )
    return paths


def _load_thumbnail_fonts(headline_px: int, badge_px: int):
    from PIL import ImageFont

    for path in _thumbnail_font_paths():
        try:
            return (
                ImageFont.truetype(path, headline_px),
                ImageFont.truetype(path, max(22, badge_px)),
            )
        except OSError:
            continue
    d = ImageFont.load_default()
    return d, d


def _wrap_headline_lines(
    text: str,
    font,
    max_width: int,
    draw,
    max_lines: int = 3,
) -> List[str]:
    """Word-wrap headline to fit width; uppercase preserved."""
    words = (text or "").upper().split()
    if not words:
        return []
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                lines.append(w[:18])
                cur = []
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    return lines[:max_lines]


def _apply_bottom_readability_gradient(img_rgba: "Image.Image", band_frac: float = 0.42) -> "Image.Image":
    """Darken bottom area with a vertical alpha ramp so white text pops."""
    from PIL import Image, ImageDraw

    w, h = img_rgba.size
    band = max(1, int(h * band_frac))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for i in range(band):
        y = h - band + i
        a = min(235, int(210 * (i + 1) / band))
        d.line([(0, y), (w, y)], fill=(0, 0, 0, a))
    return Image.alpha_composite(img_rgba, overlay)


def _mood_accent(brief: Dict) -> Tuple[str, str]:
    """Return (stroke_hex, accent_hex) from brief.color_mood."""
    mood = str(brief.get("color_mood") or "red_black").lower()
    if mood in ("cool_blue", "blue", "cinematic"):
        return "#001428", "#4fc3f7"
    if mood in ("warm", "sunset", "gold"):
        return "#2d1400", "#ffb74d"
    if mood in ("neon", "electric"):
        return "#120024", "#e040fb"
    return "#000000", "#ffffff"


def _sanitize_headline_strict(brief: Dict) -> str:
    """
    Hard text rules:
    - uppercase only
    - 3-4 words
    - no hashtags / punctuation noise
    """
    raw = (
        brief.get("selected_headline")
        or (brief.get("headline_options") or [""])[0]
        or ""
    )
    txt = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in str(raw))
    words = [w for w in txt.upper().split() if w]
    weak_fillers = {"GOES", "INSANE", "THING", "STUFF", "COOL", "VIBES"}
    words = [w for w in words if w not in weak_fillers]
    # Remove duplicates while preserving order to avoid mushy headlines.
    deduped: List[str] = []
    for w in words:
        if w not in deduped:
            deduped.append(w)
    words = deduped
    if len(words) > 4:
        words = words[:4]
    if len(words) < 3:
        mood = str(brief.get("color_mood") or "").lower()
        if "neon" in mood:
            fill = ["ALERT", "NOW", "LIVE"]
        elif "dark" in mood or "cinematic" in mood:
            fill = ["NIGHT", "DRIVE", "AHEAD"]
        else:
            fill = ["WATCH", "THIS", "NOW"]
        words.extend(fill[: max(0, 3 - len(words))])
    return " ".join(words[:4]).strip()


def _choose_style_variant(
    brief: Dict,
    platform: str,
    base_path: Path,
    headline: str,
    nonce: int = 0,
) -> Dict[str, str]:
    """
    Generate high-variety style combos from a large combinatorial set.
    Deterministic per upload/frame/headline + nonce to avoid repetitive outputs.
    """
    salt = os.environ.get("THUMB_STYLE_SALT", "")
    seed_src = f"{base_path.name}|{platform}|{headline}|{brief.get('color_mood','')}|{nonce}|{salt}"
    h = hashlib.sha256(seed_src.encode("utf-8")).hexdigest()
    pick = int(h[:12], 16)

    packs = [
        {"name": "neon_green", "stroke": "#001b00", "accent": "#39ff14", "badge": "#e53935"},
        {"name": "warning_red", "stroke": "#220000", "accent": "#ff3d00", "badge": "#fdd835"},
        {"name": "gold_premium", "stroke": "#2b1600", "accent": "#ffca28", "badge": "#1a1a1a"},
        {"name": "electric_blue", "stroke": "#001428", "accent": "#4fc3f7", "badge": "#e53935"},
        {"name": "violet_punch", "stroke": "#160026", "accent": "#e040fb", "badge": "#fdd835"},
    ]
    pack = packs[pick % len(packs)]
    accent_kinds = ["ring", "arrow", "chart_burst"]
    return {
        "pack_name": pack["name"],
        "stroke": pack["stroke"],
        "accent": pack["accent"],
        "badge_bg": pack["badge"],
        "accent_kind": accent_kinds[(pick // 7) % len(accent_kinds)],
        "stroke_w": str(3 + ((pick // 31) % 3)),
        "gradient_frac": f"{0.38 + ((pick // 101) % 9) * 0.02:.2f}",
        "sat_boost": f"{1.10 + ((pick // 211) % 7) * 0.08:.2f}",
        "con_boost": f"{1.06 + ((pick // 401) % 6) * 0.07:.2f}",
        "bri_boost": f"{1.04 + ((pick // 601) % 5) * 0.06:.2f}",
    }


def _style_signature(style: Dict[str, str], headline: str) -> str:
    core = "|".join(
        [
            str(style.get("pack_name", "")),
            str(style.get("accent_kind", "")),
            str(style.get("stroke_w", "")),
            str(headline.strip().upper()),
        ]
    )
    return hashlib.sha1(core.encode("utf-8")).hexdigest()[:20]


def _compute_pack_entropy(packs: List[str]) -> float:
    """Normalized entropy in [0,1] over recent style packs."""
    vals = [p for p in packs if p]
    if not vals:
        return 0.0
    counts: Dict[str, int] = {}
    for p in vals:
        counts[p] = counts.get(p, 0) + 1
    n = float(len(vals))
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log(p, 2) for p in probs if p > 0.0)
    h_max = math.log(max(1, len(counts)), 2)
    if h_max <= 0:
        return 0.0
    return max(0.0, min(1.0, h / h_max))


def _detect_focal_point(img_rgb: "Image.Image") -> Tuple[float, float, float]:
    """
    Approximate focal object center from edge-energy map.
    Returns (x_norm, y_norm, focal_strength_ratio).
    """
    from PIL import ImageFilter, ImageStat

    s = img_rgb.convert("L").resize((320, 180))
    edges = s.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size
    win_w, win_h = 92, 56
    best = (-1.0, 0, 0)
    global_mean = float(ImageStat.Stat(edges).mean[0] or 1.0)
    px = edges.load()
    for y in range(0, h - win_h, 8):
        for x in range(0, w - win_w, 8):
            acc = 0
            for yy in range(y, y + win_h, 4):
                for xx in range(x, x + win_w, 4):
                    acc += px[xx, yy]
            sample_n = max(1, (win_w // 4) * (win_h // 4))
            score = float(acc) / sample_n
            # small center bias so we do not lock to corners
            cx = x + win_w * 0.5
            cy = y + win_h * 0.5
            center_bias = 1.0 - (abs(cx - w * 0.5) / w) * 0.15 - (abs(cy - h * 0.5) / h) * 0.10
            score *= max(0.75, center_bias)
            if score > best[0]:
                best = (score, x, y)
    if best[0] <= 0:
        return 0.5, 0.45, 1.0
    bx, by = best[1], best[2]
    nx = (bx + win_w * 0.5) / w
    ny = (by + win_h * 0.5) / h
    return max(0.08, min(0.92, nx)), max(0.08, min(0.92, ny)), float(best[0] / max(1.0, global_mean))


def _crop_with_focal(img_rgba: "Image.Image", target_w: int, target_h: int, nx: float, ny: float) -> "Image.Image":
    """Subject-first crop: preserve focal object, not geometric center."""
    from PIL import Image

    iw, ih = img_rgba.size
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img2 = img_rgba.resize((nw, nh), Image.Resampling.LANCZOS)

    # Keep focal point slightly above center to leave room for text plate.
    fx, fy = nx * nw, ny * nh
    anchor_y = 0.43
    x0 = int(fx - target_w * 0.50)
    y0 = int(fy - target_h * anchor_y)
    x0 = max(0, min(x0, max(0, nw - target_w)))
    y0 = max(0, min(y0, max(0, nh - target_h)))
    return img2.crop((x0, y0, x0 + target_w, y0 + target_h))


def _clamp_focal_for_safe_center(nx: float, safe_center_pct: float) -> float:
    """
    Keep focal x within center-safe corridor for platforms that 1:1-crop vertical covers.
    Example: 60% safe center -> x in [0.20, 0.80].
    """
    try:
        pct = float(safe_center_pct)
    except (TypeError, ValueError):
        pct = 60.0
    pct = max(35.0, min(90.0, pct))
    half = (pct / 100.0) * 0.5
    lo = max(0.05, 0.5 - half)
    hi = min(0.95, 0.5 + half)
    return max(lo, min(hi, nx))


def _score_frame_visual_quality(image_path: Path) -> float:
    """
    Visual quality proxy in [0,1] using contrast/saturation and center saliency.
    """
    try:
        from PIL import Image, ImageFilter, ImageStat

        img = Image.open(image_path).convert("RGB")
        small = img.resize((320, 180))
        lum = small.convert("L")
        lum_std = float(ImageStat.Stat(lum).stddev[0] or 0.0)  # contrast
        sat_mean = float(ImageStat.Stat(small.convert("HSV")).mean[1] or 0.0)

        edges = lum.filter(ImageFilter.FIND_EDGES)
        w, h = edges.size
        px = edges.load()
        cx0, cx1 = int(w * 0.30), int(w * 0.70)
        cy0, cy1 = int(h * 0.22), int(h * 0.78)
        center_acc = 0.0
        outer_acc = 0.0
        center_n = 0
        outer_n = 0
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                v = float(px[x, y])
                if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                    center_acc += v
                    center_n += 1
                else:
                    outer_acc += v
                    outer_n += 1
        center_mean = center_acc / max(1, center_n)
        outer_mean = outer_acc / max(1, outer_n)
        center_ratio = center_mean / max(1.0, outer_mean)

        contrast_norm = max(0.0, min(1.0, lum_std / 64.0))
        sat_norm = max(0.0, min(1.0, sat_mean / 150.0))
        center_norm = max(0.0, min(1.0, (center_ratio - 0.75) / 0.9))
        return 0.45 * contrast_norm + 0.30 * sat_norm + 0.25 * center_norm
    except (OSError, ValueError, TypeError, ZeroDivisionError) as e:
        logger.debug("thumbnail_stage._score_frame_visual_quality: %s", e)
        return 0.0


def _auto_relight(img_rgba: "Image.Image", style: Dict[str, str]) -> "Image.Image":
    """Lift shadows + boost local contrast/saturation for dark clips."""
    from PIL import ImageEnhance, ImageFilter, ImageStat

    base_rgb = img_rgba.convert("RGB")
    lum_mean = float(ImageStat.Stat(base_rgb.convert("L")).mean[0] or 0)
    dark = lum_mean < 74

    b = float(style.get("bri_boost", "1.08"))
    c = float(style.get("con_boost", "1.10"))
    s = float(style.get("sat_boost", "1.18"))
    if dark:
        b += 0.14
        c += 0.16
        s += 0.12

    out = ImageEnhance.Brightness(base_rgb).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    out = ImageEnhance.Color(out).enhance(s)
    out = out.filter(ImageFilter.UnsharpMask(radius=2.2, percent=130, threshold=3))
    return out.convert("RGBA")


def _composition_pop_score(img_rgb: "Image.Image", focal_strength: float, headline_words: int) -> Tuple[float, Dict[str, float]]:
    """Reject low-pop outputs before save."""
    from PIL import ImageStat

    g = img_rgb.convert("L")
    stat_g = ImageStat.Stat(g)
    lum_mean = float(stat_g.mean[0] or 0)
    lum_std = float(stat_g.stddev[0] or 0)

    hsv = img_rgb.convert("HSV")
    sat_mean = float(ImageStat.Stat(hsv).mean[1] or 0)

    # score tuned to favor readable contrast + saturation + focal confidence + concise text.
    text_fit = 1.0 if 3 <= headline_words <= 5 else 0.0
    score = (
        min(35.0, lum_std * 0.9)
        + min(28.0, sat_mean * 0.22)
        + min(25.0, max(0.0, focal_strength - 0.8) * 14.0)
        + (12.0 * text_fit)
    )
    return score, {
        "lum_mean": round(lum_mean, 2),
        "lum_std": round(lum_std, 2),
        "sat_mean": round(sat_mean, 2),
        "focal_strength": round(float(focal_strength), 3),
        "headline_words": float(headline_words),
    }


def _passes_template_qa(
    img_rgb: "Image.Image",
    text_box: Optional[Tuple[int, int, int, int]],
    text_area_ratio: float,
    accent_kind: str,
    focal_strength: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Hard QA gate for styled thumbnails.
    Reject noisy/low-readability compositions so renderer re-tries style variants.
    """
    from PIL import ImageStat

    max_text_ratio = float(os.environ.get("THUMB_QA_MAX_TEXT_AREA_RATIO", "0.14") or 0.14)
    min_luma_std = float(os.environ.get("THUMB_QA_MIN_LUMA_STD", "23.0") or 23.0)
    min_text_region_std = float(os.environ.get("THUMB_QA_MIN_TEXT_REGION_STD", "28.0") or 28.0)
    min_focal_strength = float(os.environ.get("THUMB_QA_MIN_FOCAL_STRENGTH", "0.92") or 0.92)

    gray = img_rgb.convert("L")
    global_std = float(ImageStat.Stat(gray).stddev[0] or 0.0)
    text_std = global_std
    if text_box:
        x0, y0, x1, y1 = text_box
        x0 = max(0, min(img_rgb.width - 1, x0))
        y0 = max(0, min(img_rgb.height - 1, y0))
        x1 = max(x0 + 1, min(img_rgb.width, x1))
        y1 = max(y0 + 1, min(img_rgb.height, y1))
        crop = gray.crop((x0, y0, x1, y1))
        text_std = float(ImageStat.Stat(crop).stddev[0] or 0.0)

    # Arrows are intentionally constrained when focal certainty is weak.
    arrow_penalty = 0.0
    if accent_kind in ("arrow", "arrow_up", "arrow_right") and focal_strength < 1.0:
        arrow_penalty = 0.05

    ok = (
        text_area_ratio <= (max_text_ratio - arrow_penalty)
        and global_std >= min_luma_std
        and text_std >= min_text_region_std
        and focal_strength >= min_focal_strength
    )
    return ok, {
        "text_area_ratio": round(float(text_area_ratio), 4),
        "global_luma_std": round(float(global_std), 3),
        "text_region_std": round(float(text_std), 3),
        "focal_strength": round(float(focal_strength), 3),
    }


def _draw_focal_accent(
    draw,
    target_w: int,
    target_h: int,
    nx: float,
    ny: float,
    style: Dict[str, str],
    explicit: str = "",
) -> None:
    """Draw ring/arrow/chart burst anchored near focal center."""
    accent = style.get("accent", "#ffffff")
    stroke_w = max(3, min(7, int(style.get("stroke_w", "4"))))
    kind = explicit or style.get("accent_kind", "ring")

    cx = int(nx * target_w)
    cy = int(ny * target_h)
    cx = max(60, min(target_w - 60, cx))
    cy = max(60, min(target_h - 60, cy))

    if kind in ("circle", "ring", "glow_box"):
        r = max(44, int(min(target_w, target_h) * 0.06))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent, width=stroke_w)
    elif kind in ("arrow_up", "arrow_right", "arrow"):
        # Arrow points toward focal point.
        sx, sy = cx - 90, cy + 70
        draw.line([(sx, sy), (cx, cy)], fill=accent, width=stroke_w)
        draw.polygon([(cx, cy), (cx - 20, cy + 12), (cx - 12, cy - 20)], fill=accent)
    else:
        # chart_burst: jagged growth line near focal
        x0 = max(14, cx - 120)
        y0 = min(target_h - 20, cy + 90)
        pts = [(x0, y0), (x0 + 45, y0 - 22), (x0 + 84, y0 - 10), (x0 + 125, y0 - 44), (x0 + 170, y0 - 30)]
        draw.line(pts, fill=accent, width=stroke_w)
        ax, ay = pts[-1]
        draw.polygon([(ax + 18, ay - 2), (ax + 2, ay - 12), (ax + 4, ay + 6)], fill=accent)


def _render_template_thumbnail(
    base_path: Path,
    brief: Dict,
    platform: str,
    output_path: Path,
) -> bool:
    """
    Render headline + badge + directional cue onto base frame using PIL.

    Bridge for “Photoshop-level” polish without Adobe: gradient plate, multi-line type,
    mood accents, rounded badge.

    Platform-aware: YouTube 16:9 (1280x720), IG/FB 9:16 (720x1280).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow not installed — skipping template thumbnail render")
        return False

    plan = brief.get("platform_plan", {}).get(platform, {})
    if not plan.get("enabled", True):
        return False

    if platform == "youtube":
        target_w, target_h = 1280, 720
    else:
        target_w, target_h = 720, 1280  # 9:16

    try:
        source = Image.open(base_path).convert("RGBA")
    except Exception as e:
        logger.warning(f"Template render: could not load/crop base image: {e}")
        return False

    headline = _sanitize_headline_strict(brief)
    best_img = None
    best_score = -1.0
    qa_rejections: List[Dict[str, float]] = []

    avoid_map = (brief.get("_avoid_style_signatures") or {}) if isinstance(brief, dict) else {}
    avoid_set = set(avoid_map.get(platform) or [])
    recent_packs_map = (brief.get("_recent_style_packs") or {}) if isinstance(brief, dict) else {}
    recent_packs = [str(x).lower() for x in (recent_packs_map.get(platform) or []) if x]
    pack_counts: Dict[str, int] = {}
    for p in recent_packs:
        pack_counts[p] = pack_counts.get(p, 0) + 1
    pack_recent_limit = int(os.environ.get("THUMB_STYLE_PACK_REPEAT_LIMIT", "2") or 2)
    entropy_floor = float(os.environ.get("THUMB_STYLE_ENTROPY_FLOOR", "0.72") or 0.72)
    recent_window = int(os.environ.get("THUMB_STYLE_ENTROPY_WINDOW", "30") or 30)
    base_entropy = _compute_pack_entropy(recent_packs[:recent_window])

    for nonce in range(10):
        style = _choose_style_variant(brief, platform, base_path, headline, nonce=nonce)
        _apply_user_platform_colors_to_style(brief, platform, style)
        sig = _style_signature(style, headline)
        if sig in avoid_set:
            continue
        style_pack = str(style.get("pack_name") or "").lower()
        if style_pack and pack_counts.get(style_pack, 0) >= pack_recent_limit:
            # Hard cooldown on overused style packs in recent history.
            continue
        # Soft guard: if entropy is already weak, prefer introducing a less-used pack.
        if base_entropy < entropy_floor and style_pack:
            min_use = min(pack_counts.values()) if pack_counts else 0
            if pack_counts.get(style_pack, 0) > min_use:
                continue
        nx, ny, focal_strength = _detect_focal_point(source.convert("RGB"))
        safe_center_pct = float(plan.get("safe_center_pct", 60) or 60) if platform != "youtube" else 100.0
        if platform != "youtube":
            nx = _clamp_focal_for_safe_center(nx, safe_center_pct)
        img = _crop_with_focal(source, target_w, target_h, nx, ny)
        img = _auto_relight(img, style)

        safe_margin = int(min(target_w, target_h) * (0.055 if platform == "youtube" else 0.065))
        text_max_w = target_w - 2 * safe_margin
        headline_px = 72 if target_w >= 1000 else 52
        badge_px = 34 if target_w >= 1000 else 28
        font_large, font_badge = _load_thumbnail_fonts(headline_px, badge_px)
        draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        for _ in range(10):
            lines = _wrap_headline_lines(headline, font_large, text_max_w, draw_probe, max_lines=3)
            if lines:
                bb = draw_probe.textbbox((0, 0), "\n".join(lines), font=font_large)
                if bb[2] - bb[0] <= text_max_w + 6:
                    break
            if headline_px <= 34:
                break
            headline_px -= 4
            badge_px = max(22, badge_px - 2)
            font_large, font_badge = _load_thumbnail_fonts(headline_px, badge_px)
            draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        lines = _wrap_headline_lines(headline, font_large, text_max_w, draw_probe, max_lines=2 if platform != "youtube" else 3)
        # Keep text dominant but not overpowering frame area.
        text_area_ratio = 0.0
        for _ in range(8):
            if not lines:
                break
            joined_probe = "\n".join(lines)
            bb = draw_probe.textbbox((0, 0), joined_probe, font=font_large)
            tw = max(1, bb[2] - bb[0])
            th = max(1, bb[3] - bb[1])
            area_ratio = (tw * th) / float(target_w * target_h)
            text_area_ratio = area_ratio
            max_ratio = 0.14 if platform == "youtube" else 0.11
            if area_ratio <= max_ratio:
                break
            if headline_px <= 30:
                break
            headline_px -= 3
            badge_px = max(20, badge_px - 1)
            font_large, font_badge = _load_thumbnail_fonts(headline_px, badge_px)
            lines = _wrap_headline_lines(headline, font_large, text_max_w, draw_probe, max_lines=2 if platform != "youtube" else 3)

        img = _apply_bottom_readability_gradient(img, band_frac=float(style.get("gradient_frac", "0.44")))
        draw = ImageDraw.Draw(img)

        # Badge
        badge_text = (brief.get("badge_text") or "").upper()[:14]
        if badge_text:
            pad = 10
            bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
            bw = bbox[2] - bbox[0] + pad * 2
            bh = bbox[3] - bbox[1] + pad * 2
            bx0, by0 = safe_margin, safe_margin
            bx1, by1 = bx0 + bw, by0 + bh
            try:
                draw.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=style.get("badge_bg", "#e53935"), outline="#ffffff", width=2)
            except Exception:
                draw.rectangle([bx0, by0, bx1, by1], fill=style.get("badge_bg", "#e53935"), outline="#ffffff", width=2)
            draw.text((bx0 + pad, by0 + pad), badge_text, fill="#ffffff", font=font_badge)

        # Headline block
        text_box: Optional[Tuple[int, int, int, int]] = None
        if lines:
            joined = "\n".join(lines)
            bbox = draw.textbbox((0, 0), joined, font=font_large)
            block_w = bbox[2] - bbox[0]
            block_h = bbox[3] - bbox[1]
            ty0 = target_h - safe_margin - block_h - int(target_h * 0.02)
            tx0 = (target_w - block_w) // 2
            text_box = (tx0 - 12, ty0 - 10, tx0 + block_w + 12, ty0 + block_h + 10)
            off = 0
            sw = max(2, int(style.get("stroke_w", "4")))
            for line in lines:
                lb = draw.textbbox((0, 0), line, font=font_large)
                lw = lb[2] - lb[0]
                lx = (target_w - lw) // 2
                ly = ty0 + off
                for dx, dy in ((-sw, -sw), (-sw, sw), (sw, -sw), (sw, sw), (-sw, 0), (sw, 0), (0, -sw), (0, sw)):
                    draw.text((lx + dx, ly + dy), line, fill=style.get("stroke", "#000000"), font=font_large)
                draw.text((lx, ly), line, fill="#ffffff", font=font_large)
                off += (lb[3] - lb[1]) + 6

        explicit_elem = str(brief.get("directional_element") or "").strip().lower()
        if explicit_elem not in ("none", "off", "false", "no"):
            _draw_focal_accent(draw, target_w, target_h, nx, ny, style, explicit=explicit_elem)

        qa_ok, qa_meta = _passes_template_qa(
            img.convert("RGB"),
            text_box=text_box,
            text_area_ratio=text_area_ratio,
            accent_kind=explicit_elem or style.get("accent_kind", ""),
            focal_strength=focal_strength,
        )
        if not qa_ok:
            if len(qa_rejections) < 12:
                rec: Dict[str, float] = {
                    "nonce": float(nonce),
                    "focal_strength": round(float(focal_strength), 3),
                }
                for k, v in qa_meta.items():
                    try:
                        rec[k] = float(v)
                    except Exception:
                        continue
                qa_rejections.append(rec)
            logger.debug("thumb-template qa-reject platform=%s meta=%s", platform, qa_meta)
            continue

        score, meta = _composition_pop_score(img.convert("RGB"), focal_strength, len(headline.split()))
        if score > best_score:
            best_score = score
            best_img = img.convert("RGB")
            brief.setdefault("_render_meta", {})[platform] = {
                "signature": sig,
                "style_pack": style.get("pack_name", ""),
                "score": round(float(score), 3),
                "entropy_before": round(base_entropy, 4),
                "qa": qa_meta,
            }
            if qa_rejections:
                brief.setdefault("_qa_rejections", {})[platform] = qa_rejections[:]
            logger.debug(
                "thumb-template pop-score=%.2f meta=%s style=%s",
                score,
                meta,
                style.get("pack_name"),
            )

        # Hard quality gate: pass early if strong composition.
        if score >= 56.0 and focal_strength >= 1.10:
            try:
                img.convert("RGB").save(output_path, "JPEG", quality=92, subsampling=0)
                return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE
            except Exception:
                continue

    # Final fallback: only accept if composition is still above minimum threshold.
    if best_img is None or best_score < 43.0:
        logger.warning(f"Template render rejected by composition gate (best={best_score:.2f})")
        return False
    try:
        best_img.save(output_path, "JPEG", quality=92, subsampling=0)
        return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE
    except Exception as e:
        logger.warning(f"Template render save failed: {e}")
        return False


# ============================================================
# AI Image Edit (optional premium — with guardrails + fallback)
# ============================================================

async def _ai_edit_thumbnail(
    base_path: Path,
    brief: Dict,
    output_path: Path,
    retry_reduce: bool = False,
) -> bool:
    """
    Use OpenAI image edit to add headline/badge/props to base frame.
    Guardrails: no logos, no new objects except arrow/badge/simple props.
    Max 2 retries; falls back to template in caller.
    Note: OpenAI edits API may return base64 or URL; we handle both.
    """
    if not OPENAI_API_KEY:
        return False
    instruction = (
        f"Add these elements to the image. Headline text (ALL CAPS): {brief.get('selected_headline', '')}. "
        f"Badge: {brief.get('badge_text', '')}. "
        f"Add a {brief.get('directional_element', 'circle')} highlight. "
        "No new objects except: arrow, badge, simple prop icons. "
        "No logos, no brand marks, no watermarks. "
        "Keep key elements centered for mobile crop. "
    )
    if retry_reduce:
        instruction += "Reduce text size, increase contrast, fewer effects. "
    try:
        with open(base_path, "rb") as f:
            image_data = f.read()
        async with httpx.AsyncClient(timeout=120) as client:
            # Multipart: `model` must be form fields (data=), not only in files= — httpx/OpenAI
            # otherwise the API returns missing_required_parameter model.
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data={
                    "model": OPENAI_IMAGE_EDIT_MODEL,
                    "prompt": instruction,
                    "size": "1024x1024",
                },
                files={"image": ("frame.jpg", image_data, "image/jpeg")},
            )
            if response.status_code != 200:
                logger.warning(f"AI thumbnail edit HTTP {response.status_code}: {response.text[:200]}")
                return False
            data = response.json()
            items = data.get("data", [])
            if not items:
                return False
            item = items[0]
            if "b64_json" in item:
                out_bytes = base64.b64decode(item["b64_json"])
            elif "url" in item:
                dl = await client.get(item["url"])
                if dl.status_code != 200:
                    return False
                out_bytes = dl.content
            else:
                return False
        output_path.write_bytes(out_bytes)
        return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE
    except Exception as e:
        logger.warning(f"AI thumbnail edit failed: {e}")
        return False


async def _maybe_composite_viral_background(
    ctx: JobContext,
    frame_path: Path,
    brief: Dict,
    platform: str,
    category: str,
) -> Optional[Path]:
    """
    rembg/remove.bg subject isolation + fal/replicate AI background + composite.
    Gated by can_ai_thumbnail_styling and THUMB_BG_COMPOSITE_ENABLED.
    """
    if str(os.environ.get("THUMB_BG_COMPOSITE_ENABLED", "true")).lower() != "true":
        return None
    if not ctx.entitlements or not getattr(ctx.entitlements, "can_ai_thumbnail_styling", False):
        return None
    plan = brief.get("platform_plan", {}).get(platform, {})
    if not plan.get("enabled", True):
        return None
    if platform == "youtube":
        w, h = 1280, 720
    else:
        w, h = 720, 1280

    from .background_gen import (
        build_background_prompt,
        composite_subject_on_background,
        generate_ai_background,
        generate_kontext_background_replicate,
        isolate_subject,
    )

    ac = ctx.audio_context or {}
    mood = str(ac.get("thumbnail_mood") or ac.get("mood") or "neon_vibrant")
    emotion = str(
        ac.get("emotion")
        or ac.get("dominant_emotion")
        or ac.get("emotional_tone")
        or ""
    )
    headline = str(brief.get("selected_headline") or "")[:80]
    extra = ""
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict):
        extra = (vu.get("scene_description") or "")[:500]

    try:
        subject = await isolate_subject(frame_path, ctx.temp_dir)
        if not subject:
            return None
        prompt = build_background_prompt(category, mood, emotion, headline, extra_context=extra)
        bg_path = None
        if os.environ.get("REPLICATE_KONTEXT_MODEL", "").strip():
            bg_path = await generate_kontext_background_replicate(
                frame_path, prompt, w, h, ctx.temp_dir
            )
        if not bg_path:
            bg_path = await generate_ai_background(prompt, w, h, ctx.temp_dir)
        if not bg_path:
            return None
        comp = composite_subject_on_background(subject, bg_path, w, h, anchor="right", scale=0.82)
        out = ctx.temp_dir / f"thumb_composite_{platform}_{ctx.upload_id}.jpg"
        comp.save(out, "JPEG", quality=92)
        if out.exists() and out.stat().st_size >= MIN_THUMB_SIZE:
            logger.info("[thumbnail] bg composite ok platform=%s -> %s", platform, out.name)
            return out
    except Exception as e:
        logger.warning("[thumbnail] bg composite failed (non-fatal): %s", e)
    return None


async def _try_playwright_html_thumbnail(
    ctx: JobContext,
    frame_path: Path,
    brief: Dict,
    platform: str,
    out_path: Path,
) -> bool:
    """
    HTML/CSS thumbnail via headless Chromium (playwright_stage).
    Gated by THUMB_HTML_RENDER_ENABLED and PLAYWRIGHT_ENABLED.
    """
    if str(os.environ.get("THUMB_HTML_RENDER_ENABLED", "true")).lower() != "true":
        return False
    try:
        from .playwright_stage import PLAYWRIGHT_AVAILABLE, PLAYWRIGHT_ENABLED, render_template
    except ImportError:
        return False
    if not PLAYWRIGHT_AVAILABLE or not PLAYWRIGHT_ENABLED:
        return False
    plan = brief.get("platform_plan", {}).get(platform, {})
    if not plan.get("enabled", True):
        return False
    if platform == "youtube":
        w, h = 1280, 720
    else:
        w, h = 720, 1280
    headline = html_module.escape((brief.get("selected_headline") or "WATCH")[:80])
    subtext = html_module.escape((brief.get("notes") or "")[:120])
    badge = html_module.escape((brief.get("badge_text") or "")[:20])
    te = str(brief.get("text_effect") or "").lower().strip()
    if te == "glitch":
        tpl = "GLITCH"
    elif te == "chrome":
        tpl = "CHROME"
    elif te == "fire":
        tpl = "FIRE_SCROLL"
    elif te == "neon":
        tpl = "NEON_DROP"
    elif te == "clean":
        tpl = "CINEMATIC"
    else:
        mood_key = str(brief.get("color_mood") or "").lower()
        pal = getattr(ctx, "frame_color_palette", None)
        if isinstance(pal, dict) and pal.get("mood_hint") == "warm" and "fire" not in mood_key:
            tpl = "HEAT"
        elif "neon" in mood_key or "violet" in mood_key or "magenta" in mood_key:
            tpl = "NEON_DROP"
        elif "gold" in mood_key or "premium" in mood_key:
            tpl = "HEAT"
        elif "fire" in mood_key or "ember" in mood_key:
            tpl = "FIRE_SCROLL"
        elif "glitch" in mood_key or "rgb" in mood_key:
            tpl = "GLITCH"
        elif "chrome" in mood_key or "metal" in mood_key:
            tpl = "CHROME"
        elif "cinema" in mood_key or "film" in mood_key:
            tpl = "CINEMATIC"
        else:
            tpl = "BRIGHT_POP"
    try:
        result = await render_template(tpl, frame_path, headline, subtext, badge, w, h, out_path)
        if result and Path(out_path).exists() and out_path.stat().st_size >= MIN_THUMB_SIZE:
            logger.info("[thumbnail] Playwright HTML render ok platform=%s tpl=%s", platform, tpl)
            return True
    except Exception as e:
        logger.warning("[thumbnail] Playwright HTML render failed: %s", e)
    return False


# ============================================================
# ffprobe — get video duration
# ============================================================

async def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe. Returns 30.0 on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                dur = float(stream.get("duration", 0) or 0)
                if dur > 0:
                    return dur
    except Exception as e:
        logger.warning(f"ffprobe duration failed: {e}")
    return 30.0


# ============================================================
# Frame extraction
# ============================================================

async def _extract_frame(video_path: Path, output_path: Path, offset: float) -> bool:
    """Extract a single JPEG frame at `offset` seconds. Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=1080:-2",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if (
        proc.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size >= MIN_THUMB_SIZE
    ):
        return True

    logger.debug(
        f"FFmpeg thumb failed at {offset:.1f}s (rc={proc.returncode}): "
        f"{stderr.decode()[-200:]}"
    )
    return False


# ============================================================
# Sharpness scoring via FFmpeg blurdetect
# ============================================================

async def _score_sharpness(image_path: Path) -> float:
    """
    Run FFmpeg blurdetect on a JPEG. Returns sharpness score (0–1, higher = sharper).
    Returns 0.0 on failure (triggers file-size fallback in caller).
    """
    cmd = [
        "ffmpeg", "-i", str(image_path),
        "-vf", "blurdetect=high=0.1",
        "-f", "null", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        output = stderr.decode()

        for line in output.splitlines():
            if "blur_mean:" in line:
                for part in line.split():
                    if part.startswith("blur_mean:"):
                        blur_val = float(part.split(":")[1])
                        return max(0.0, 1.0 - blur_val)
    except Exception as e:
        logger.debug(f"blurdetect failed for {image_path.name}: {e}")
    return 0.0


# ============================================================
# Offset distribution
# ============================================================

def _distribute_offsets_spaced(
    duration: float,
    n: int,
    user_offset: Optional[float] = None,
) -> List[float]:
    """
    Generate N evenly-spaced offsets across the video duration.

    Anchors: first frame at 5% (avoids black intros), last at 90% (avoids fade-outs).
    Middle N-2 frames distributed evenly between anchors.
    n==1 uses user_offset if provided, else 30%.
    All values clamped to [0.5, duration-0.5].
    """
    if duration <= 0:
        duration = 30.0

    def clamp(v: float) -> float:
        return max(0.5, min(v, duration - 0.5))

    n = max(1, n)

    if n == 1:
        return [clamp(user_offset if user_offset is not None else duration * 0.30)]

    if n == 2:
        return [clamp(duration * 0.05), clamp(duration * 0.90)]

    start = clamp(duration * 0.05)
    end = clamp(duration * 0.90)
    middle_count = n - 2
    step = (end - start) / (middle_count + 1)
    middle = [clamp(start + step * (i + 1)) for i in range(middle_count)]

    return [start] + middle + [end]


def _distribute_offsets(
    duration: float,
    n: int,
    user_offset: Optional[float] = None,
    interval_sec: Optional[float] = None,
) -> List[float]:
    """
    Frame timestamps for thumbnail extraction.

    When ``interval_sec`` is set (>0) and n>1, offsets are placed at least
    ``interval_sec`` apart starting near 5% of duration (matches settings
    "Thumbnail Interval — seconds between frames"). Falls back to even-spaced
    anchors if fewer than two timestamps fit.
    """
    if duration <= 0:
        duration = 30.0

    def clamp(v: float) -> float:
        return max(0.5, min(v, duration - 0.5))

    n = max(1, n)

    if n == 1:
        return _distribute_offsets_spaced(duration, 1, user_offset)

    use_interval = interval_sec is not None and float(interval_sec) > 0
    if not use_interval:
        return _distribute_offsets_spaced(duration, n, user_offset)

    step = max(1.0, min(float(interval_sec), 120.0))
    usable_end = max(0.5, duration - 0.5)
    start = clamp(duration * 0.05)
    offsets: List[float] = [start]
    cur = start
    while len(offsets) < n:
        cur += step
        if cur > usable_end - 0.01:
            break
        offsets.append(clamp(cur))

    if len(offsets) < 2:
        return _distribute_offsets_spaced(duration, n, user_offset)

    return offsets[:n]


def _user_platform_colors_from_settings(us: dict) -> Dict[str, str]:
    """Badge/accent hex colors from user_preferences / user_settings (Settings → Platform Colors)."""
    out: Dict[str, str] = {}
    _camel_color = {
        "tiktok": "tiktokColor",
        "youtube": "youtubeColor",
        "instagram": "instagramColor",
        "facebook": "facebookColor",
    }
    for plat in ("tiktok", "youtube", "instagram", "facebook"):
        v = (us.get(f"{plat}_color") or us.get(_camel_color.get(plat, "")) or "").strip()
        if v.startswith("#"):
            out[plat] = v
    acc = (us.get("accent_color") or us.get("accentColor") or "").strip()
    if acc.startswith("#"):
        out["accent"] = acc
    return out


def _apply_user_platform_colors_to_style(brief: Dict, platform: str, style: Dict) -> None:
    """Override template badge (and optional accent) from saved platform colors."""
    upc = (brief.get("user_platform_colors") or {}) if isinstance(brief, dict) else {}
    key = (platform or "").lower()
    hexv = (upc.get(key) or "").strip()
    if hexv.startswith("#") and len(hexv) in (4, 5, 7, 9):
        style["badge_bg"] = hexv
    acc = (upc.get("accent") or "").strip()
    if acc.startswith("#") and len(acc) in (4, 5, 7, 9):
        style["accent"] = acc


# ============================================================
# Stage Entry Point
# ============================================================

async def run_thumbnail_stage(ctx: JobContext, db_pool=None) -> JobContext:
    """
    Generate candidate thumbnails, score for sharpness, then run AI-powered
    content-category-aware selection to pick the most engaging frame.

    Tier gating:
      - Number of candidates exposed = ctx.entitlements.max_thumbnails
      - GPT vision frame selection + thumbnail briefs run when OPENAI_API_KEY is set
        (supercharged path; not gated on plan can_ai)
      - When OpenAI is not available: falls back cleanly to sharpest frame

    When OPENAI_API_KEY is set, internally extracts max(max_thumbnails, 4) frames
    to give the AI meaningful choices, even on single-thumbnail tiers —
    the AI picks the single best one and only that one is exposed.

    ctx fields set by this stage:
      ctx.thumbnail_path        — final best frame (AI-picked or sharpest)
      ctx.thumbnail_paths       — all candidates up to max_thumbnails
      ctx.thumbnail_scores      — {str(path): sharpness} for all extracted
      ctx.output_artifacts      — thumbnail, thumbnail_candidates,
                                  thumbnail_scores, thumbnail_category,
                                  thumbnail_selection_method
    """
    ctx.mark_stage("thumbnail")

    if not ctx.entitlements or not should_generate_thumbnails(ctx.entitlements):
        raise SkipStage("Thumbnail generation not enabled for tier")

    us0 = ctx.user_settings or {}
    _auto = us0.get("auto_thumbnails", us0.get("autoThumbnails"))
    if _auto is None:
        _auto = us0.get("auto_generate_thumbnails", us0.get("autoGenerateThumbnails", True))
    if not _auto:
        raise SkipStage("Auto thumbnail generation disabled in settings")

    # ── Source video ────────────────────────────────────────────────────────
    video_path: Optional[Path] = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break

    if not video_path:
        raise SkipStage("No video file available for thumbnail generation")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available")

    # ── Tier gates ──────────────────────────────────────────────────────────
    max_thumbnails = 1
    if ctx.entitlements:
        max_thumbnails = max(1, int(getattr(ctx.entitlements, "max_thumbnails", 1) or 1))

    ai_key_present = bool(OPENAI_API_KEY)
    # Vision/brief frame selection uses OpenAI whenever the key is set (supercharged path).
    thumb_supercharged_ai = ai_key_present

    # When OpenAI vision is available, extract at least 4 frames for meaningful selection.
    extraction_count = max(max_thumbnails, 4 if thumb_supercharged_ai else 1)

    # User-specified manual offset (single-thumbnail mode only)
    raw_offset = (ctx.user_settings or {}).get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        user_offset = float(raw_offset)
        user_offset = max(0.0, min(user_offset, MAX_THUMBNAIL_OFFSET))
    except (TypeError, ValueError):
        user_offset = DEFAULT_THUMBNAIL_OFFSET

    raw_iv = us0.get("thumbnail_interval", us0.get("thumbnailInterval", 5))
    try:
        thumb_interval_sec = float(raw_iv)
    except (TypeError, ValueError):
        thumb_interval_sec = 5.0
    thumb_interval_sec = max(1.0, min(thumb_interval_sec, 120.0))

    # ── Category detection ──────────────────────────────────────────────────
    category = _detect_category(ctx)

    try:
        await fetch_trend_intel(ctx)
    except Exception as e:
        logger.debug("[thumbnail] trend_intel skipped: %s", e)

    logger.info(
        f"Thumbnail stage: video={video_path.name}, "
        f"max_thumbnails={max_thumbnails}, extraction_count={extraction_count}, "
        f"category={category}, "
        f"ai={'supercharged' if thumb_supercharged_ai else 'no-openai-key'}"
    )

    # ── Duration probe ──────────────────────────────────────────────────────
    duration = await _get_video_duration(video_path)
    logger.debug(f"Video duration: {duration:.1f}s")

    # ── Distribute offsets ──────────────────────────────────────────────────
    offsets = _distribute_offsets(
        duration=duration,
        n=extraction_count,
        user_offset=user_offset if extraction_count == 1 else None,
        interval_sec=thumb_interval_sec if extraction_count > 1 else None,
    )
    try:
        ctx.output_artifacts["thumbnail_interval_seconds"] = str(int(thumb_interval_sec))
    except Exception:
        pass
    logger.debug(f"Thumbnail offsets: {[f'{o:.1f}s' for o in offsets]}")

    # ── Extract and score all frames ────────────────────────────────────────
    candidates: List[Tuple[Path, float]] = []
    component_scores: Dict[str, Dict[str, float]] = {}
    path_to_offset: Dict[str, float] = {}

    for idx, offset in enumerate(offsets):
        out_path = ctx.temp_dir / f"thumb_{ctx.upload_id}_{idx:02d}.jpg"
        success = await _extract_frame(video_path, out_path, offset)

        if not success and offset > 0:
            logger.debug(f"Frame at {offset:.1f}s failed — retrying at t=0")
            success = await _extract_frame(video_path, out_path, 0.0)

        if success:
            path_to_offset[str(out_path)] = float(offset)
            sharpness = await _score_sharpness(out_path)
            if sharpness == 0.0:
                sharpness = min(1.0, out_path.stat().st_size / 1_000_000)  # file-size proxy
            visual = _score_frame_visual_quality(out_path)
            # Strong source frame selection: sharpness + clear subject + contrast/saturation.
            score = (0.58 * sharpness) + (0.42 * visual)
            component_scores[str(out_path)] = {
                "sharpness": round(float(sharpness), 4),
                "visual": round(float(visual), 4),
                "combined": round(float(score), 4),
            }
            candidates.append((out_path, score))
            logger.debug(
                f"  Frame {idx}: {out_path.name} @ {offset:.1f}s — "
                f"sharpness={sharpness:.4f}, visual={visual:.4f}, combined={score:.4f}, size={out_path.stat().st_size // 1024}KB"
            )
        else:
            logger.warning(f"  Frame {idx} @ {offset:.1f}s failed — skipping")

    if not candidates:
        logger.warning("Thumbnail generation failed for all offsets (non-fatal)")
        raise SkipStage("FFmpeg thumbnail extraction produced no output")

    async def _ml_pick_thumbnail_ai_vs_sharpness() -> tuple[bool, dict]:
        """
        Decide whether to run the AI frame selection pass.
        Uses per-user empirical strategy performance from `upload_quality_scores_daily`.
        """
        try:
            if not db_pool:
                return False, {"reason": "no_db_pool"}
            user_id = str(getattr(ctx, "user_id", "") or "")
            if not user_id:
                return False, {"reason": "no_user_id"}

            lookback_days = int(os.environ.get("ML_THUMBNAIL_BIAS_LOOKBACK_DAYS", os.environ.get("ML_SCORING_LOOKBACK_DAYS", "30")) or 30)
            lookback_days = max(7, min(lookback_days, 180))

            category_key = str(category or "general").strip().lower().replace("-", "_")
            ai_key = f"ai_{category_key}"
            sharp_key = "sharpness"

            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        strategy_key,
                        SUM(samples)::bigint AS samples,
                        CASE
                            WHEN SUM(samples) > 0 THEN
                                SUM(mean_engagement * samples::double precision)
                                / NULLIF(SUM(samples::double precision), 0)
                            ELSE 0.0
                        END AS w_mean_engagement
                    FROM upload_quality_scores_daily
                    WHERE user_id = $1::uuid
                      AND platform = 'all'
                      AND strategy_key = ANY($2::text[])
                      AND day >= (CURRENT_DATE - ($3::int || ' days')::interval)::date
                    GROUP BY strategy_key
                    """,
                    user_id,
                    [sharp_key, ai_key],
                    lookback_days,
                )

            by_key: Dict[str, Dict[str, float]] = {}
            for r in rows or []:
                sk = (r.get("strategy_key") or "").strip().lower()
                by_key[sk] = {
                    "samples": float(r.get("samples") or 0),
                    "w_mean_engagement": float(r.get("w_mean_engagement") or 0.0),
                }

            sharp_row = by_key.get(sharp_key) or {}
            ai_row = by_key.get(ai_key) or {}
            min_ai = int(os.environ.get("ML_THUMBNAIL_MIN_SAMPLES_AI", "4") or 4)
            min_sharp = int(os.environ.get("ML_THUMBNAIL_MIN_SAMPLES_SHARP", "2") or 2)
            margin = float(os.environ.get("ML_THUMBNAIL_EB_MARGIN", "0.12") or 0.12)

            decision, detail = prefer_ai_thumbnail_vs_sharpness(
                sharp_mean=sharp_row.get("w_mean_engagement"),
                sharp_samples=int(sharp_row.get("samples") or 0),
                ai_mean=ai_row.get("w_mean_engagement"),
                ai_samples=int(ai_row.get("samples") or 0),
                min_samples_ai=max(1, min_ai),
                min_samples_sharp=max(0, min_sharp),
                margin=margin,
            )
            return decision, {
                "reason": "ml_thumbnail_selection_bias",
                "category": category_key,
                "strategy_keys": {"sharpness": sharp_key, "ai": ai_key},
                "lookback_days": lookback_days,
                "aggregates": {
                    "sharpness": sharp_row,
                    "ai": ai_row,
                },
                **detail,
            }
        except Exception as e:
            return False, {"reason": "ml_decide_failed", "error": str(e)}

    # ── Sharpness-best (always computed; used as fallback) ──────────────────
    sharpness_best_path, sharpness_best_score = max(candidates, key=lambda x: x[1])

    # ── AI selection pass ───────────────────────────────────────────────────
    ai_selected_path: Optional[Path] = None
    selection_method = "sharpness"
    ml_bias = None

    if thumb_supercharged_ai and len(candidates) > 1:
        # Empirical bias: decide whether AI selection historically beats sharpness.
        use_ai = False
        ml_bias, ml_decision = None, None
        try:
            use_ai, ml_decision = await _ml_pick_thumbnail_ai_vs_sharpness()
            ml_bias = ml_decision
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("thumbnail_stage: ml thumbnail decision failed: %s", e)
            use_ai = False

        ctx.output_artifacts["_ml_thumbnail_selection_bias"] = ml_bias or {"reason": "ml_decide_skipped"}

        if use_ai:
            try:
                ai_selected_path = await _ai_select_best_frame(candidates, category, ctx)
            except Exception as e:
                logger.warning(f"AI thumbnail selection raised unexpectedly: {e}")
                ai_selected_path = None

            if ai_selected_path and Path(ai_selected_path).exists():
                selection_method = f"ai_{category}"
            else:
                logger.info(
                    f"ML-preferred AI selection returned nothing — falling back to sharpest: "
                    f"{sharpness_best_path.name}"
                )
        else:
            reason = "ml_bias_says_sharpness_better_or_no_samples"
            logger.debug(f"AI thumbnail selection skipped: {reason}")
            ai_selected_path = None
    else:
        reason = "no OPENAI_API_KEY" if not ai_key_present else "only 1 candidate"
        logger.debug(f"AI thumbnail selection skipped: {reason}")

    # ── Final best frame ────────────────────────────────────────────────────
    best_path  = ai_selected_path if ai_selected_path else sharpness_best_path
    best_score = next(
        (s for p, s in candidates if str(p) == str(best_path)),
        sharpness_best_score,
    )

    tiktok_plats = {str(p).lower() for p in (ctx.platforms or [])}
    tiktok_mode = os.environ.get("TIKTOK_COVER_FRAME_MODE", "").strip().lower()
    if not tiktok_mode:
        tiktok_mode = "motion_blur" if "tiktok" in tiktok_plats else "balanced"
    tiktok_offset_seconds, tiktok_pick_reason = pick_tiktok_cover_offset_seconds(
        candidates, path_to_offset, component_scores, tiktok_mode, best_path=best_path
    )
    try:
        ctx.output_artifacts["tiktok_cover_frame_mode"] = tiktok_mode
        ctx.output_artifacts["tiktok_cover_pick_reason"] = tiktok_pick_reason
        ctx.output_artifacts["thumbnail_frame_offsets_json"] = json.dumps(
            {Path(k).name: round(v, 3) for k, v in path_to_offset.items()}
        )
    except Exception as e:
        logger.debug("thumbnail_stage: could not write tiktok cover / frame offset artifacts: %s", e)

    # ── Populate ctx ────────────────────────────────────────────────────────
    # Chronological candidate list (caption_stage uses these for multi-frame story)
    # Respect max_thumbnails tier cap for the exposed list
    ctx.thumbnail_paths = [p for p, _ in candidates[:max_thumbnails]]
    ctx.thumbnail_scores = {str(p): s for p, s in candidates}

    # Ensure the AI-selected frame is always in the list (even if it's frame 5
    # on a single-thumbnail tier)
    if best_path not in ctx.thumbnail_paths:
        ctx.thumbnail_paths.insert(0, best_path)

    ctx.thumbnail_path = best_path

    # Dominant colors from best frame → thumbnail brief + Playwright/CSS hints
    ctx.frame_color_palette = None
    try:
        from .color_palette import extract_palette_from_image

        ctx.frame_color_palette = extract_palette_from_image(best_path)
        if ctx.frame_color_palette:
            ctx.output_artifacts["thumbnail_color_palette_json"] = json.dumps(ctx.frame_color_palette)
    except Exception as e:
        logger.debug("thumbnail_stage: extract_palette_from_image skipped: %s", e)

    # Artifacts — picked up by worker.py for R2 upload and DB save
    ctx.output_artifacts["thumbnail"]                   = str(best_path)
    ctx.output_artifacts["thumbnail_category"]          = category
    ctx.output_artifacts["thumbnail_selection_method"]  = selection_method
    ctx.output_artifacts["thumbnail_candidates"]        = json.dumps(
        [str(p) for p in ctx.thumbnail_paths]
    )
    ctx.output_artifacts["thumbnail_scores"]            = json.dumps(
        {str(p): round(s, 4) for p, s in candidates}
    )
    ctx.output_artifacts["thumbnail_score_components"]  = json.dumps(component_scores)

    # ── Styled thumbnails (MrBeast-style composite) — Trill + non-Trill, every upload ──
    # Gated by: can_custom_thumbnails + user pref styled_thumbnails (default True)
    can_custom = bool(getattr(ctx.entitlements, "can_custom_thumbnails", False) if ctx.entitlements else False)
    can_ai_style = bool(getattr(ctx.entitlements, "can_ai_thumbnail_styling", False) if ctx.entitlements else False)
    us = ctx.user_settings or {}
    styled_enabled = us.get("styled_thumbnails", us.get("styledThumbnails", True))
    if can_custom and styled_enabled and ctx.temp_dir:
        try:
            brief: Optional[Dict] = None
            if thumb_supercharged_ai:
                brief = await _generate_thumbnail_brief(ctx, category)
            if not brief and thumb_supercharged_ai:
                # Fallback brief when GPT fails — minimal defaults
                brief = {
                    "selected_headline": (ctx.get_effective_title() or "WATCH")[:20].upper(),
                    "headline_options": [],
                    "badge_text": "",
                    "badge_style": "red",
                    "directional_element": "none",
                    "props": [],
                    "emotion_cue": "excited",
                    "color_mood": "red_black",
                    "platform_plan": {
                        "youtube": {"enabled": True, "canvas": "16:9"},
                        "instagram": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
                        "facebook": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
                        "tiktok": {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
                    },
                    "notes": "Fallback brief",
                }
            elif not brief:
                brief = {
                    "selected_headline": (ctx.get_effective_title() or "WATCH")[:20].upper(),
                    "headline_options": [],
                    "badge_text": "",
                    "badge_style": "red",
                    "directional_element": "none",
                    "props": [],
                    "emotion_cue": "excited",
                    "color_mood": "red_black",
                    "platform_plan": {
                        "youtube": {"enabled": True, "canvas": "16:9"},
                        "instagram": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
                        "facebook": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
                        "tiktok": {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
                    },
                    "notes": "No AI — minimal brief",
                }
            if isinstance(brief, dict):
                brief["user_platform_colors"] = _user_platform_colors_from_settings(us)
            if brief is not None and "tiktok" in tiktok_plats:
                brief.setdefault("platform_plan", {})
                brief["platform_plan"].setdefault(
                    "tiktok",
                    {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
                )
                brief["platform_plan"]["tiktok"]["thumb_offset_seconds"] = round(
                    float(tiktok_offset_seconds), 2
                )
            ctx.output_artifacts["thumbnail_brief_json"] = json.dumps(brief)
            ctx.thumbnail_brief = brief
            prior_raw = ctx.output_artifacts.get("_recent_thumbnail_style_signatures", "{}")
            prior_map = json_dict(
                prior_raw, default={}, context="thumbnail._recent_thumbnail_style_signatures"
            )
            prior_pack_raw = ctx.output_artifacts.get("_recent_thumbnail_style_packs", "{}")
            prior_pack_map = json_dict(
                prior_pack_raw, default={}, context="thumbnail._recent_thumbnail_style_packs"
            )
            brief["_avoid_style_signatures"] = prior_map
            brief["_recent_style_packs"] = prior_pack_map

            # TikTok: no custom thumbnail via API — store thumb_offset for worker
            platform_map: Dict[str, str] = {}
            platform_engine_map: Dict[str, str] = {}
            tiktok_plan = brief.get("platform_plan", {}).get("tiktok", {})
            ctx.output_artifacts["tiktok_thumb_offset_seconds"] = str(
                tiktok_plan.get("thumb_offset_seconds", 1.5)
            )

            # Render per platform (YouTube, Instagram, Facebook)
            platforms_to_render = [p for p in ("youtube", "instagram", "facebook")
                                  if (brief.get("platform_plan", {}).get(p, {}).get("enabled", True))
                                  and p in [pl.lower() for pl in (ctx.platforms or [])]]
            render_method = "none"
            # Default to deterministic template rendering for consistency.
            # AI image edits can be flashy/inconsistent, so keep them as fallback unless explicitly preferred.
            ai_style_mode = str(os.environ.get("THUMB_AI_STYLE_MODE", "fallback")).strip().lower()
            prefer_ai_edit = ai_style_mode in ("prefer_ai", "ai_first")
            render_engine_mode = _thumbnail_render_engine_mode()
            studio_enabled = bool(us.get("thumbnail_studio_enabled", us.get("thumbnailStudioEnabled", False)))
            engine_pref_enabled = bool(
                us.get(
                    "thumbnail_studio_engine_enabled",
                    us.get("thumbnailStudioEngineEnabled", us.get("thumbnail_pikzels_enabled", us.get("thumbnailPikzelsEnabled", False))),
                )
            )
            use_engine_this_upload = bool(
                us.get(
                    "thumbnail_use_studio_engine",
                    us.get("thumbnailUseStudioEngine", us.get("thumbnail_use_pikzels", us.get("thumbnailUsePikzels", False))),
                )
            )
            allow_persona = bool(us.get("thumbnail_persona_enabled", us.get("thumbnailPersonaEnabled", False)))
            use_persona_this_upload = bool(us.get("thumbnail_use_persona", us.get("thumbnailUsePersona", False)))
            persona_id = str(us.get("thumbnail_persona_id", us.get("thumbnailPersonaId", "")) or "").strip()
            try:
                persona_strength = int(us.get("thumbnail_persona_strength", us.get("thumbnailPersonaStrength", 70)) or 70)
            except (TypeError, ValueError):
                persona_strength = 70
            persona_strength = max(0, min(100, persona_strength))
            persona_payload: Optional[Dict[str, Any]] = None
            if studio_enabled and allow_persona and use_persona_this_upload and persona_id and db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        prow = await conn.fetchrow(
                            """
                            SELECT id, name, profile_json
                            FROM creator_personas
                            WHERE id = $1::uuid AND user_id = $2::uuid
                            """,
                            persona_id,
                            str(ctx.user_id),
                        )
                        if prow:
                            irows = await conn.fetch(
                                """
                                SELECT image_url
                                FROM creator_persona_images
                                WHERE persona_id = $1::uuid
                                ORDER BY created_at ASC
                                LIMIT 5
                                """,
                                persona_id,
                            )
                            persona_payload = {
                                "id": str(prow.get("id") or ""),
                                "name": str(prow.get("name") or ""),
                                "profile": dict(prow.get("profile_json") or {}),
                                "strength": persona_strength,
                                "reference_images": [str(r.get("image_url") or "") for r in (irows or []) if str(r.get("image_url") or "").strip()],
                            }
                except Exception as e:
                    logger.debug("[thumbnail] persona fetch failed: %s", e)

            allow_studio_renderer = (
                studio_enabled
                and engine_pref_enabled
                and use_engine_this_upload
                and render_engine_mode in ("studio", "auto")
                and studio_renderer_enabled()
            )
            ctx.output_artifacts["thumbnail_persona_applied"] = "true" if persona_payload else "false"
            primary_styled: Optional[Path] = None  # Prefer YouTube for primary
            youtube_frame_src: Optional[Path] = None
            for platform in platforms_to_render:
                out_name = f"thumb_styled_{platform}_{ctx.upload_id}.jpg"
                out_path = ctx.temp_dir / out_name
                ok = False
                engine_used = "internal"
                frame_src = best_path
                if can_ai_style:
                    try:
                        comp_path = await _maybe_composite_viral_background(
                            ctx, best_path, brief, platform, category
                        )
                        if comp_path and comp_path.exists():
                            frame_src = comp_path
                    except Exception as e:
                        logger.debug("[thumbnail] bg composite not used: %s", e)

                if not ok and allow_studio_renderer:
                    ok = await render_thumbnail_with_studio_renderer(
                        frame_src,
                        brief,
                        platform,
                        out_path,
                        upload_id=str(ctx.upload_id or ""),
                        category=str(category or ""),
                        persona=persona_payload,
                        options={
                            "thumbnail_studio_enabled": studio_enabled,
                            "persona_enabled": bool(persona_payload),
                            "persona_strength": persona_strength,
                        },
                    )
                    if ok:
                        engine_used = "studio_renderer"
                        render_method = "studio_renderer"
                    elif render_engine_mode == "studio":
                        logger.info(
                            "[thumbnail] studio render unavailable for %s; fallback to internal engine",
                            platform,
                        )

                if prefer_ai_edit and can_ai_style and OPENAI_API_KEY:
                    ok = await _ai_edit_thumbnail(frame_src, brief, out_path, retry_reduce=False)
                    if not ok:
                        ok = await _ai_edit_thumbnail(frame_src, brief, out_path, retry_reduce=True)
                    if ok:
                        render_method = "ai_edit"
                if not ok and can_ai_style:
                    ok = await _try_playwright_html_thumbnail(ctx, frame_src, brief, platform, out_path)
                    if ok:
                        render_method = "playwright_html"
                if not ok:
                    ok = _render_template_thumbnail(frame_src, brief, platform, out_path)
                    if ok:
                        render_method = "template"
                if not ok and (not prefer_ai_edit) and can_ai_style and OPENAI_API_KEY:
                    ok = await _ai_edit_thumbnail(frame_src, brief, out_path, retry_reduce=False)
                    if not ok:
                        ok = await _ai_edit_thumbnail(frame_src, brief, out_path, retry_reduce=True)
                    if ok:
                        render_method = "ai_edit"
                if ok:
                    platform_map[platform] = str(out_path)
                    platform_engine_map[platform] = engine_used
                    if primary_styled is None or platform == "youtube":
                        primary_styled = out_path
                    if platform == "youtube":
                        youtube_frame_src = frame_src
            if primary_styled and primary_styled.exists():
                ctx.thumbnail_path = primary_styled
                ctx.output_artifacts["thumbnail"] = str(primary_styled)

            ctx.output_artifacts["thumbnail_render_method"] = render_method
            ctx.output_artifacts["thumbnail_render_engine"] = (
                "studio_renderer"
                if ("studio_renderer" in platform_engine_map.values() or "pikzels" in platform_engine_map.values())
                else "internal"
            )
            ctx.output_artifacts["platform_thumbnail_engine_map"] = json.dumps(platform_engine_map)
            ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(platform_map)
            # YouTube search-result size legibility (168×94 proxy)
            try:
                yt_local = platform_map.get("youtube")
                if YOUTUBE_SEARCH_PREVIEW_QA and yt_local and Path(yt_local).exists():
                    qa_prev = assess_youtube_search_preview_readability(Path(yt_local))
                    ctx.output_artifacts["youtube_search_preview_qa_json"] = json.dumps(qa_prev)
            except Exception as e:
                logger.debug("[thumbnail] youtube preview QA skipped: %s", e)
            # Extra YouTube JPEGs for Studio "Test & Compare" (manual — API has no thumbnailTests)
            ab_extra = max(0, min(2, int(os.environ.get("YOUTUBE_THUMBNAIL_AB_EXTRA", "0") or 0)))
            ab_enabled = os.environ.get("YOUTUBE_THUMBNAIL_AB_ENABLED", "false").lower() == "true"
            if (
                ab_enabled
                and ab_extra > 0
                and "youtube" in platform_map
                and youtube_frame_src
                and isinstance(brief, dict)
            ):
                opts = [str(x).strip() for x in (brief.get("headline_options") or []) if str(x).strip()]
                main_h = str(brief.get("selected_headline") or "WATCH")[:80]
                pool = ([main_h] + opts)[:6]
                effects = ["glitch", "neon", "chrome"]
                ab_list = []
                for i in range(ab_extra):
                    vb = copy.deepcopy(brief)
                    alt = pool[(i + 1) % len(pool)] if len(pool) > 1 else main_h
                    vb["selected_headline"] = alt[:80]
                    vb["text_effect"] = effects[i % len(effects)]
                    out_ab = ctx.temp_dir / f"thumb_styled_youtube_ab{i}_{ctx.upload_id}.jpg"
                    ok_ab = False
                    if prefer_ai_edit and can_ai_style and OPENAI_API_KEY:
                        ok_ab = await _ai_edit_thumbnail(youtube_frame_src, vb, out_ab, retry_reduce=False)
                        if not ok_ab:
                            ok_ab = await _ai_edit_thumbnail(youtube_frame_src, vb, out_ab, retry_reduce=True)
                    if not ok_ab and can_ai_style:
                        ok_ab = await _try_playwright_html_thumbnail(
                            ctx, youtube_frame_src, vb, "youtube", out_ab
                        )
                    if not ok_ab:
                        ok_ab = _render_template_thumbnail(youtube_frame_src, vb, "youtube", out_ab)
                    if not ok_ab and (not prefer_ai_edit) and can_ai_style and OPENAI_API_KEY:
                        ok_ab = await _ai_edit_thumbnail(youtube_frame_src, vb, out_ab, retry_reduce=False)
                        if not ok_ab:
                            ok_ab = await _ai_edit_thumbnail(youtube_frame_src, vb, out_ab, retry_reduce=True)
                    if ok_ab:
                        ab_list.append(
                            {
                                "path": str(out_ab),
                                "label": f"B{i + 1}",
                                "headline": alt[:80],
                                "text_effect": vb.get("text_effect"),
                            }
                        )
                if ab_list:
                    ctx.output_artifacts["youtube_thumbnail_ab_candidates"] = json.dumps(ab_list)
                    ctx.output_artifacts["youtube_thumbnail_ab_note"] = (
                        "Public YouTube Data API v3 supports thumbnails.set only. "
                        "Use Studio Test & Compare with these extra JPEGs (R2 keys in worker)."
                    )
            style_meta = (brief.get("_render_meta") or {}) if isinstance(brief, dict) else {}
            if isinstance(style_meta, dict) and style_meta:
                ctx.output_artifacts["thumbnail_style_signatures"] = json.dumps(style_meta)
                qa_rejections = (brief.get("_qa_rejections") or {}) if isinstance(brief, dict) else {}
                if isinstance(qa_rejections, dict) and qa_rejections:
                    try:
                        ctx.output_artifacts["thumbnail_qa_rejections"] = json.dumps(qa_rejections)
                    except Exception as e:
                        logger.debug("thumbnail_stage: could not serialize thumbnail_qa_rejections: %s", e)
                # Emit audit block for enterprise governance / dashboards.
                try:
                    policy = {}
                    for p, md in style_meta.items():
                        if not isinstance(md, dict):
                            continue
                        policy[p] = {
                            "style_pack": str(md.get("style_pack") or ""),
                            "signature": str(md.get("signature") or ""),
                            "score": float(md.get("score") or 0.0),
                            "entropy_before": float(md.get("entropy_before") or 0.0),
                            "pack_repeat_limit": int(os.environ.get("THUMB_STYLE_PACK_REPEAT_LIMIT", "2") or 2),
                            "entropy_floor": float(os.environ.get("THUMB_STYLE_ENTROPY_FLOOR", "0.72") or 0.72),
                        }
                    ctx.output_artifacts["thumbnail_style_policy"] = json.dumps(policy)
                except Exception as e:
                    logger.debug("thumbnail_stage: could not build thumbnail_style_policy artifact: %s", e)
        except Exception as e:
            logger.warning(f"[thumbnail] Styled thumbnail pipeline failed (non-fatal): {e}")

    logger.info(
        f"Thumbnail stage complete: {len(candidates)} frames, "
        f"selected={best_path.name} "
        f"(method={selection_method}, sharpness={best_score:.4f})"
    )

    return ctx
