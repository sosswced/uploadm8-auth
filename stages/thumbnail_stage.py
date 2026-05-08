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
     for the content type, not just the sharpest). Skipped when Thumbnail
     Studio + Pikzels will render first (sharpest frame is enough as compositor input).
  7. Set ctx.thumbnail_path  = AI-selected (or sharpest as fallback)
     Set ctx.thumbnail_paths = all candidates in chronological order
     Set ctx.thumbnail_scores = {str(path): score} for all candidates
  8. Store metadata in ctx.output_artifacts for queue UI display
  9. [When can_custom_thumbnails] Generate Thumbnail Brief (JSON) via GPT,
     render MrBeast-style composite (headline, badge, arrow) per platform:
     YouTube 16:9, Instagram/Facebook 9:16 center-safe, TikTok thumb_offset only.
     Render via template (PIL) or AI image edit (when can_ai_thumbnail_styling).

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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.content_attribution import (
    normalize_thumbnail_render_pipeline,
    normalize_thumbnail_selection_mode,
)
from core.thumbnail_text import (
    CATEGORY_HEADLINE_FALLBACKS,
    clean_thumbnail_headline,
    is_generic_thumbnail_headline,
)

from .context import JobContext, THUMBNAIL_BRIEF_PROMPT
from .errors import SkipStage, ThumbnailError
from .ai_service_costs import user_pref_ai_service_enabled
from .ffmpeg_env import resolve_ffmpeg_executable
from services.hydration_payload import apply_hydration_payload_to_thumbnail_brief
from services.thumbnail_studio import closeness_to_pikzels_image_weight
from services.thumbnail_trace import trace_append

from .pikzels_api import (
    generate_pikzels_text_brief,
    refine_thumbnail_with_pikzels_edit,
    render_thumbnail_with_studio_renderer,
    studio_renderer_enabled,
)

logger = logging.getLogger("uploadm8-worker.thumbnail")

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_THUMBNAIL_OFFSET = 1.0
MAX_THUMBNAIL_OFFSET     = 300.0
MIN_THUMB_SIZE           = 2048          # bytes — smaller = rejected
OPENAI_API_KEY           = os.environ.get("OPENAI_API_KEY", "")
OPENAI_THUMB_MODEL       = os.environ.get("OPENAI_THUMB_MODEL", "gpt-4o-mini")
STRICT_STUDIO_ENV        = os.environ.get("UPLOADM8_THUMBNAIL_STRICT_STUDIO", "").strip().lower() in ("1", "true", "yes", "on")


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
    Multi-signal content category detection.
    Layer 1: user-provided caption/title hints
    Layer 2: filename keyword scan
    Layer 3: vision/audio/video-understanding/telemetry signals
    Layer 4: fall back to 'general' (GPT identifies from frames in the prompt)
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

    for text in (ctx.caption, ctx.title):
        result = _scan(text or "")
        if result:
            logger.debug(f"Thumbnail category from user hint: {result}")
            return result

    result = _scan(ctx.filename or "")
    if result:
        logger.debug(f"Thumbnail category from filename: {result}")
        return result

    evidence: List[str] = []
    vu = ctx.video_understanding or {}
    if isinstance(vu, dict):
        for key in ("scene_description", "description", "title_suggestion", "summary"):
            value = vu.get(key)
            if isinstance(value, str) and value.strip():
                evidence.append(value)

    vi = ctx.video_intelligence_context or {}
    if isinstance(vi, dict):
        for key in ("labels", "label_names", "shot_labels", "objects", "ocr_text"):
            value = vi.get(key)
            if isinstance(value, list):
                evidence.extend(str(x.get("description") if isinstance(x, dict) else x) for x in value[:24])
            elif isinstance(value, str):
                evidence.append(value)

    vc = ctx.vision_context or {}
    if isinstance(vc, dict):
        for key in ("label_names", "landmark_names", "logo_names", "ocr_text"):
            value = vc.get(key)
            if isinstance(value, list):
                evidence.extend(str(x) for x in value[:24])
            elif isinstance(value, str):
                evidence.append(value)

    ac = ctx.audio_context or {}
    if isinstance(ac, dict):
        for key in ("suggested_keywords", "yamnet_events", "top_sound_class", "sound_profile", "music_title", "music_artist", "music_genre", "fusion_narrative"):
            value = ac.get(key)
            if isinstance(value, list):
                evidence.extend(str(x) for x in value[:24])
            elif isinstance(value, str):
                evidence.append(value)

    tel = ctx.telemetry_data or ctx.telemetry
    if tel and ((getattr(tel, "max_speed_mph", 0) or 0) > 0 or getattr(tel, "location_road", None)):
        evidence.append("dashcam road drive speed automotive route")

    result = _scan(" ".join(x for x in evidence if x))
    if result:
        logger.debug(f"Thumbnail category from fused context: {result}")
        return result

    logger.debug("Thumbnail category: general (GPT will identify from frames)")
    return "general"


# ============================================================
# Thumbnail Brief Guardrails
# ============================================================

def _first_list_value(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    for nested_key in ("name", "description", "text", "label"):
                        nested = item.get(nested_key)
                        if isinstance(nested, str) and nested.strip():
                            return nested.strip()
        elif isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _concrete_thumbnail_headline(ctx: JobContext, category: str) -> str:
    """Build a truthful fallback headline from known context, never generic hype."""
    cat = (category or "general").strip().lower()
    tel = ctx.telemetry_data or ctx.telemetry
    if tel:
        mph = getattr(tel, "max_speed_mph", 0) or 0
        if cat in {"automotive", "travel", "sports"} and mph and float(mph) >= 10:
            return f"{float(mph):.0f} MPH RUN"
        road = (getattr(tel, "location_road", None) or "").strip()
        if cat in {"automotive", "travel"} and road:
            cleaned = clean_thumbnail_headline(f"{road} drive", max_words=4)
            if cleaned:
                return cleaned

    osd = ctx.dashcam_osd_context or {}
    if isinstance(osd, dict):
        mph = osd.get("max_speed_mph") or osd.get("peak_speed_mph")
        try:
            if cat in {"automotive", "travel", "sports"} and mph and float(mph) >= 10:
                return f"{float(mph):.0f} MPH RUN"
        except (TypeError, ValueError):
            pass

    vc = ctx.vision_context or {}
    if isinstance(vc, dict):
        concrete = _first_list_value(vc, "landmark_names", "logo_names", "labels", "label_names", "objects")
        if concrete:
            cleaned = clean_thumbnail_headline(concrete, max_words=4)
            if cleaned and not is_generic_thumbnail_headline(cleaned):
                return cleaned
        ocr = str(vc.get("ocr_text") or "").strip()
        if ocr:
            cleaned = clean_thumbnail_headline(ocr, max_words=4)
            if cleaned and not is_generic_thumbnail_headline(cleaned):
                return cleaned

    ac = ctx.audio_context or {}
    if isinstance(ac, dict):
        music = ac.get("music_title") or ac.get("track_title") or ac.get("title")
        artist = ac.get("music_artist") or ac.get("artist")
        if music or artist:
            cleaned = clean_thumbnail_headline(" ".join(str(x) for x in (artist, music) if x), max_words=4)
            if cleaned and not is_generic_thumbnail_headline(cleaned):
                return cleaned

    for source in (ctx.get_effective_title(), ctx.get_effective_caption(), ctx.filename):
        cleaned = clean_thumbnail_headline(source, max_words=5)
        if cleaned and not is_generic_thumbnail_headline(cleaned):
            return cleaned

    return CATEGORY_HEADLINE_FALLBACKS.get(category, CATEGORY_HEADLINE_FALLBACKS["general"])


def _default_platform_plan() -> Dict[str, Dict[str, Any]]:
    return {
        "youtube": {"enabled": True, "canvas": "16:9"},
        "instagram": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
        "facebook": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
        "tiktok": {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
    }


def _sanitize_thumbnail_brief(ctx: JobContext, brief: Optional[Dict[str, Any]], category: str, *, note: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = dict(brief or {})
    brief_vars = ctx.get_thumbnail_brief_vars(category=category)
    for src, dst in (
        ("geo_context", "geo_context"),
        ("osd_context", "osd_context"),
        ("trill_context", "trill_context"),
        ("music_context", "music_context"),
        ("speech_context", "speech_context"),
        ("signal_hashtags", "signal_hashtags"),
        ("fusion_summary", "fusion_summary"),
        ("hydration_story", "hydration_story"),
    ):
        v = str(brief_vars.get(src) or "").strip()
        if v and not str(out.get(dst) or "").strip():
            out[dst] = v

    fallback = _concrete_thumbnail_headline(ctx, category)
    selected = clean_thumbnail_headline(out.get("selected_headline"), max_words=5)
    if is_generic_thumbnail_headline(selected):
        selected = fallback
    out["selected_headline"] = selected

    options: List[str] = []
    raw_options = out.get("headline_options") or []
    if isinstance(raw_options, list):
        for item in raw_options:
            candidate = item.get("text") if isinstance(item, dict) else item
            cleaned = clean_thumbnail_headline(candidate, max_words=5)
            if cleaned and not is_generic_thumbnail_headline(cleaned) and cleaned not in options:
                options.append(cleaned)
    for candidate in (selected, fallback, CATEGORY_HEADLINE_FALLBACKS.get(category, "VIDEO HIGHLIGHT")):
        cleaned = clean_thumbnail_headline(candidate, max_words=5)
        if cleaned and cleaned not in options:
            options.append(cleaned)
    out["headline_options"] = options[:3]

    badge = clean_thumbnail_headline(out.get("badge_text"), max_words=2)[:14]
    tel = ctx.telemetry_data or ctx.telemetry
    osd = ctx.dashcam_osd_context or {}
    osd_speed = 0.0
    if isinstance(osd, dict):
        try:
            osd_speed = float(osd.get("max_speed_mph") or osd.get("peak_speed_mph") or 0)
        except (TypeError, ValueError):
            osd_speed = 0.0
    has_speed = bool((tel and (getattr(tel, "max_speed_mph", 0) or 0) > 0) or osd_speed > 0)
    if badge in {"NEW", "FAST", "SPEED"} and not has_speed:
        badge = ""
    out["badge_text"] = badge
    if out.get("badge_style") not in {"red", "yellow", "white", "black"}:
        out["badge_style"] = "red"
    if out.get("directional_element") not in {"arrow_up", "arrow_right", "circle", "glow_box"}:
        out["directional_element"] = "circle"
    if out.get("emotion_cue") not in {"shocked", "excited", "serious", "laughing"}:
        out["emotion_cue"] = "serious"
    if out.get("color_mood") not in {"red_black", "blue_black", "gold_black", "neon"}:
        out["color_mood"] = "red_black"
    if not isinstance(out.get("props"), list):
        out["props"] = []

    plan = _default_platform_plan()
    raw_plan = out.get("platform_plan")
    if isinstance(raw_plan, dict):
        for platform, defaults in plan.items():
            if isinstance(raw_plan.get(platform), dict):
                merged = dict(defaults)
                merged.update(raw_plan[platform])
                plan[platform] = merged
    out["platform_plan"] = plan
    if note:
        out["notes"] = note
    return out


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
        except Exception as e:
            logger.debug(f"Could not attach frame {path.name}: {e}")

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

        parsed = json.loads(answer)
        selected_idx = int(parsed.get("selected_frame", 0))
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

    except json.JSONDecodeError as e:
        logger.warning(f"AI thumbnail response not valid JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"AI thumbnail selection failed (non-fatal, using sharpness): {e}")
        return None


# ============================================================
# Thumbnail Brief Generator (platform-aware JSON)
# ============================================================

async def _generate_thumbnail_brief(ctx: JobContext, category: str) -> Optional[Dict]:
    """
    Generate a platform-aware Thumbnail Brief (JSON) using GPT.
    Returns parsed brief dict or None on failure.
    """
    if not OPENAI_API_KEY or not ctx.entitlements or not ctx.entitlements.can_ai:
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
        return _sanitize_thumbnail_brief(ctx, brief, category)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Thumbnail brief parse failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Thumbnail brief generation failed: {e}")
        return None


# ============================================================
# Template Renderer (PIL overlays — deterministic fallback)
# ============================================================

def _render_template_thumbnail(
    base_path: Path,
    brief: Dict,
    platform: str,
    output_path: Path,
) -> bool:
    """
    Render headline + badge + directional element onto base frame using PIL.
    Platform-aware: YouTube 16:9 (1280x720), IG/FB 9:16 center-safe (720x1280).
    Returns True on success.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed — skipping template thumbnail render")
        return False

    plan = brief.get("platform_plan", {}).get(platform, {})
    if not plan.get("enabled", True):
        return False

    canvas = plan.get("canvas", "16:9")
    if platform == "youtube":
        target_w, target_h = 1280, 720
    else:
        target_w, target_h = 720, 1280  # 9:16

    try:
        img = Image.open(base_path).convert("RGB")
        iw, ih = img.size
        # Scale/crop to target aspect
        scale = max(target_w / iw, target_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x0, y0 = (nw - target_w) // 2, (nh - target_h) // 2
        img = img.crop((x0, y0, x0 + target_w, y0 + target_h))
    except Exception as e:
        logger.warning(f"Template render: could not load/crop base image: {e}")
        return False

    draw = ImageDraw.Draw(img)
    safe_margin = int(min(target_w, target_h) * 0.06)

    # Font — try common paths, fallback to default
    font_large = font_badge = None
    for path in (
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            font_large = ImageFont.truetype(path, 72)
            font_badge = ImageFont.truetype(path, 36)
            break
        except OSError:
            continue
    if font_large is None:
        font_large = ImageFont.load_default()
        font_badge = font_large

    headline = clean_thumbnail_headline(
        brief.get("selected_headline") or brief.get("headline_options", [""])[0],
        max_words=5,
        max_chars=30,
    )
    if is_generic_thumbnail_headline(headline):
        headline = ""
    badge_text = (brief.get("badge_text") or "").upper()[:12]

    # Badge colors
    badge_style = brief.get("badge_style", "red")
    badge_colors = {"red": "#e53935", "yellow": "#fdd835", "white": "#ffffff", "black": "#1a1a1a"}
    badge_fg = "#ffffff" if badge_style in ("red", "black") else "#1a1a1a"
    badge_bg = badge_colors.get(badge_style, "#e53935")

    # Draw badge (top-left)
    if badge_text:
        pad = 8
        bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
        bw, bh = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
        draw.rectangle([safe_margin, safe_margin, safe_margin + bw, safe_margin + bh], fill=badge_bg, outline="#fff", width=2)
        draw.text((safe_margin + pad, safe_margin + pad), badge_text, fill=badge_fg, font=font_badge)

    # Headline (bottom, centered)
    if headline:
        bbox = draw.textbbox((0, 0), headline, font=font_large)
        tw = bbox[2] - bbox[0]
        tx = (target_w - tw) // 2
        ty = target_h - safe_margin - (bbox[3] - bbox[1]) - 20
        # Stroke for readability
        for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
            draw.text((tx+dx, ty+dy), headline, fill="#000", font=font_large)
        draw.text((tx, ty), headline, fill="#fff", font=font_large)

    # Simple directional element (circle highlight — bottom-right area)
    elem = brief.get("directional_element", "circle")
    if elem in ("circle", "glow_box"):
        cx, cy = target_w - safe_margin - 80, target_h - safe_margin - 80
        draw.ellipse([cx-50, cy-50, cx+50, cy+50], outline="#fff", width=4)
    elif elem in ("arrow_up", "arrow_right"):
        cx, cy = target_w - safe_margin - 60, target_h - safe_margin - 60
        draw.polygon([(cx, cy-30), (cx-20, cy+20), (cx+20, cy+20)], outline="#fff", width=3)

    try:
        img.save(output_path, "JPEG", quality=90)
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
    headline = clean_thumbnail_headline(brief.get("selected_headline"), max_words=5, max_chars=34)
    if is_generic_thumbnail_headline(headline):
        headline = ""
    text_instruction = (
        f"Headline text (ALL CAPS): {headline}. "
        if headline
        else "Do not add headline text; no generic overlay phrases. "
    )
    instruction = (
        f"Add these elements to the image. {text_instruction}"
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
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={
                    "image": ("frame.jpg", image_data, "image/jpeg"),
                    "prompt": (None, instruction),
                    "size": (None, "1024x1024"),
                },
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


# ============================================================
# ffprobe — get video duration
# ============================================================

async def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe. Returns 30.0 on failure."""
    cmd = [
        resolve_ffmpeg_executable("ffprobe") or "ffprobe", "-v", "quiet",
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
    _ffmpeg = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        _ffmpeg, "-y",
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
    _ffmpeg = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        _ffmpeg, "-i", str(image_path),
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

def _distribute_offsets(
    duration: float,
    n: int,
    user_offset: Optional[float] = None,
    min_gap_seconds: Optional[float] = None,
) -> List[float]:
    """
    Generate N offsets across the video duration.

    When ``min_gap_seconds`` is set (>0) and n>1, place frames starting near 5% duration
    stepping by that many seconds (clamped), matching Upload Preferences "Thumbnail interval".

    Otherwise: evenly-spaced anchors (5% … 90%) with distributed middles.
    n==1 uses user_offset if provided, else 30%. All values clamped to [0.5, duration-0.5].
    """
    if duration <= 0:
        duration = 30.0

    def clamp(v: float) -> float:
        return max(0.5, min(v, duration - 0.5))

    n = max(1, n)

    if n == 1:
        return [clamp(user_offset if user_offset is not None else duration * 0.30)]

    gap = float(min_gap_seconds) if min_gap_seconds is not None else 0.0
    if gap > 0 and n > 1:
        start = clamp(duration * 0.05)
        end_limit = clamp(duration * 0.95)
        pts: List[float] = []
        t = start
        while len(pts) < n and t <= end_limit + 1e-6:
            pts.append(clamp(t))
            t += gap
        if len(pts) >= n:
            return pts[:n]

    if n == 2:
        return [clamp(duration * 0.05), clamp(duration * 0.90)]

    start = clamp(duration * 0.05)
    end = clamp(duration * 0.90)
    middle_count = n - 2
    step = (end - start) / (middle_count + 1)
    middle = [clamp(start + step * (i + 1)) for i in range(middle_count)]

    return [start] + middle + [end]


def _thumbnail_styled_render_order(
    pipeline_pref: str,
    *,
    studio_ok: bool,
    ai_edit_ok: bool,
) -> List[str]:
    """
    Per-platform compositing attempts. Values: studio | ai_edit | template.
    Mirrors ML attribution keys (thumbnailRenderPipeline).
    """
    p = (pipeline_pref or "auto").lower().strip()
    if p not in ("auto", "studio_renderer", "ai_edit", "template", "none"):
        p = "auto"
    if p == "none":
        return []
    # When Pikzels/Studio is available we ALWAYS try it first, but we MUST keep
    # ai_edit/template behind it so that a transient Pikzels failure (network
    # blip, 4xx, empty response) still produces a styled cover instead of
    # leaving the raw extracted frame as the published thumbnail. The previous
    # behaviour of returning ["studio"] alone is what caused dashcam frames
    # with burned-in OSD overlays to ship as final thumbnails.
    if studio_ok:
        chain: List[str] = ["studio"]
        if ai_edit_ok:
            chain.append("ai_edit")
        chain.append("template")
        return chain
    if p == "template":
        return ["template"]
    if p == "ai_edit":
        order: List[str] = []
        if ai_edit_ok:
            order.append("ai_edit")
        order.append("template")
        return order
    out: List[str] = []
    if ai_edit_ok:
        out.append("ai_edit")
    out.append("template")
    return out


def pikzels_studio_eligible_for_styled_thumbnail(
    us: Dict[str, Any],
    entitlements: Any,
    *,
    require_auto_thumbnails: bool = True,
) -> bool:
    """
    Whether the Pikzels v2 studio renderer should run for styled thumbnails.

    ``require_auto_thumbnails=False`` is for explicit user actions (e.g. API
    "generate thumbnail") where auto-thumbnail scheduling prefs must not block Pikzels.
    """
    if require_auto_thumbnails and not (us.get("auto_thumbnails") or us.get("autoThumbnails")):
        return False
    render_pipeline_pref = normalize_thumbnail_render_pipeline(us)
    if render_pipeline_pref == "none":
        return False
    can_custom = bool(getattr(entitlements, "can_custom_thumbnails", False) if entitlements else False)
    styled_enabled = us.get("styled_thumbnails", us.get("styledThumbnails", True))
    if not (can_custom and styled_enabled):
        return False
    designer_on = user_pref_ai_service_enabled(us, "thumbnail_ai", default=True)
    ready = studio_renderer_enabled()

    def _opt_in_default_true(*keys: str) -> bool:
        for k in keys:
            if k in us:
                return bool(us.get(k))
        return ready

    studio_flow = _opt_in_default_true("thumbnail_studio_enabled", "thumbnailStudioEnabled")
    use_studio_engine = _opt_in_default_true(
        "thumbnail_studio_engine_enabled",
        "thumbnailStudioEngineEnabled",
        "thumbnail_pikzels_enabled",
        "thumbnailPikzelsEnabled",
    )
    return bool(designer_on and studio_flow and use_studio_engine and ready)


def _studio_persona_for_request(us: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return Pikzels persona/style payload plus prompt options for the studio API.

    Persona/style auto-enable rule:
      A persona or style UUID being saved on the user is itself an *implicit*
      opt-in to use it for this render. We only honor an explicit opt-out
      (``thumbnail_persona_enabled = False``). This stops the silent failure
      mode where users save a default persona but never flip the toggle, and
      every published thumbnail comes back unstyled.
    """
    try:
        strength = int(us.get("thumbnail_persona_strength") or us.get("thumbnailPersonaStrength") or 70)
    except (TypeError, ValueError):
        strength = 70
    strength = max(0, min(100, strength))
    opts: Dict[str, Any] = {"persona_strength": strength}

    style_hint = (
        us.get("thumbnail_style")
        or us.get("thumbnailStyle")
        or us.get("thumbnail_style_prompt")
        or us.get("thumbnailStylePrompt")
        or ""
    )
    if str(style_hint or "").strip():
        opts["style_hint"] = str(style_hint).strip()[:180]

    strategy = _thumbnail_default_strategy(us)
    if strategy:
        parts: List[str] = []
        if strategy.get("layout_name") or strategy.get("layout_pattern"):
            parts.append(
                f"{str(strategy.get('layout_name') or '').strip()} {str(strategy.get('layout_pattern') or '').strip()}".strip()
            )
        if strategy.get("audience_niche"):
            parts.append(f"audience {str(strategy.get('audience_niche')).replace('_', ' ')}")
        if strategy.get("competitor_gap_mode"):
            parts.append("competitor-gap variation")
        if parts:
            merged_hint = "; ".join(parts)
            if opts.get("style_hint"):
                merged_hint = f"{opts['style_hint']}; {merged_hint}"
            opts["style_hint"] = merged_hint[:180]
        rs = strategy.get("reference_strength")
        if isinstance(rs, (int, float)):
            try:
                opts["image_weight"] = closeness_to_pikzels_image_weight(int(rs))
            except (TypeError, ValueError):
                pass

    pkz_persona = str(
        us.get("thumbnail_pikzels_persona_id")
        or us.get("thumbnailPikzelsPersonaId")
        or ""
    ).strip()

    # Loud warning for the silent-fail mode: user saved a default persona in
    # Settings but ``merge_pikzels_thumbnail_persona_id`` couldn't resolve a
    # linked Pikzels Pikzonality UUID for it (no ``pikzels_user_assets`` row,
    # status != linked, or NULL ``pikzels_pikzonality_id``). Without this log
    # the persona simply never gets applied and thumbnails keep returning
    # unstyled — exactly the experience the user reported.
    default_pid = str(
        us.get("thumbnail_default_persona_id")
        or us.get("thumbnailDefaultPersonaId")
        or ""
    ).strip()
    if default_pid and not pkz_persona:
        logger.warning(
            "[thumb-renderer] user has thumbnail_default_persona_id=%s but no linked "
            "Pikzels Pikzonality UUID was resolved (merge_pikzels_thumbnail_persona_id "
            "found no row with status='linked' in pikzels_user_assets). Pikzels render "
            "will run WITHOUT the user's saved persona. Re-link the persona in the "
            "Thumbnail Studio or call /api/thumbnail-studio/personas/<id>/sync-pikzels.",
            default_pid,
        )

    persona_pref_explicit = (
        "thumbnail_persona_enabled" in us
        or "thumbnailPersonaEnabled" in us
    )
    persona_explicit_disable = persona_pref_explicit and not bool(
        us.get("thumbnail_persona_enabled")
        if "thumbnail_persona_enabled" in us
        else us.get("thumbnailPersonaEnabled")
    )
    if pkz_persona and not persona_explicit_disable:
        return {"id": pkz_persona, "kind": "persona"}, opts

    style_pkz = str(
        us.get("thumbnail_pikzels_style_id")
        or us.get("thumbnailPikzelsStyleId")
        or us.get("thumbnail_style_pikzels_id")
        or us.get("thumbnailStylePikzelsId")
        or ""
    ).strip()
    style_pref_explicit = (
        "thumbnail_style_enabled" in us
        or "thumbnailStyleEnabled" in us
        or "thumbnail_pikzels_style_enabled" in us
        or "thumbnailPikzelsStyleEnabled" in us
    )
    style_explicit_disable = False
    if style_pref_explicit:
        for k in (
            "thumbnail_style_enabled",
            "thumbnailStyleEnabled",
            "thumbnail_pikzels_style_enabled",
            "thumbnailPikzelsStyleEnabled",
        ):
            if k in us:
                style_explicit_disable = not bool(us.get(k))
                break
    if style_pkz and not style_explicit_disable:
        return {"id": style_pkz, "kind": "style"}, opts

    return None, opts


def _thumbnail_default_strategy(us: Dict[str, Any]) -> Dict[str, Any]:
    raw = us.get("thumbnail_studio_default_strategy")
    if raw is None:
        raw = us.get("thumbnailStudioDefaultStrategy")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _apply_thumbnail_default_strategy(
    brief: Dict[str, Any],
    us: Dict[str, Any],
    *,
    category: str,
) -> Dict[str, Any]:
    """
    Apply the user's selected Studio variation as a durable upload-time strategy.

    The strategy supplies layout/audience/reference rules. Hydration from the current
    upload still fills speech/music/geo/vision details, so outputs stay consistent in
    structure but adapt to each video.
    """
    strategy = _thumbnail_default_strategy(us)
    if not strategy:
        return brief
    b = dict(brief or {})
    bits: List[str] = []
    layout_name = str(strategy.get("layout_name") or "").strip()
    layout_pattern = str(strategy.get("layout_pattern") or "").strip()
    if layout_name or layout_pattern:
        bits.append(
            f"Default layout strategy: {layout_name} {layout_pattern}".strip()[:260]
        )
    audience = str(strategy.get("audience_niche") or category or "").strip()
    if audience:
        bits.append(f"Audience/niche: {audience.replace('_', ' ')}."[:160])
    if strategy.get("competitor_gap_mode"):
        bits.append("Differentiate from common competitor thumbnails while preserving the selected layout family.")
    pos = str(strategy.get("text_position") or "").strip()
    contrast = str(strategy.get("contrast_profile") or "").replace("_", " ").strip()
    emotion = str(strategy.get("emotion") or "").strip()
    visual_bits = [x for x in (emotion and f"{emotion} emotion", pos and f"{pos} text", contrast and f"{contrast} contrast") if x]
    if visual_bits:
        bits.append("Visual style: " + ", ".join(visual_bits) + ".")
    if bits:
        existing = str(b.get("notes") or "").strip()
        joined = " ".join(bits)
        b["notes"] = (existing + " " + joined).strip()[:700]
        b["default_strategy"] = strategy
    return b


def _strict_studio_mode_enabled(us: Dict[str, Any]) -> bool:
    """Fail fast when Studio isn't available or cannot render a platform cover."""
    if STRICT_STUDIO_ENV:
        return True
    raw = us.get("thumbnail_studio_strict")
    if raw is None:
        raw = us.get("thumbnailStudioStrict")
    if raw is None:
        raw = us.get("thumbnail_pikzels_strict")
    if raw is None:
        raw = us.get("thumbnailPikzelsStrict")
    return bool(raw)


def _hydration_pikzels_edit_enabled() -> bool:
    v = (os.environ.get("THUMBNAIL_HYDRATION_PIKZELS_EDIT") or "1").strip().lower()
    return v not in ("0", "false", "off", "no", "disabled")


def _upload_requests_thumbnail_render(us: Dict[str, Any]) -> bool:
    """
    True when upload-scoped prefs explicitly request thumbnail rendering.

    Presign can snapshot per-upload overrides like ``thumbnail_use_pikzels`` /
    ``thumbnail_use_studio_engine`` into ``uploads.user_preferences``. Those
    jobs should still render thumbnails even if account-level auto_thumbnails is
    disabled.
    """
    for key in (
        "thumbnail_pikzels_enabled",
        "thumbnailPikzelsEnabled",
        "thumbnail_studio_engine_enabled",
        "thumbnailStudioEngineEnabled",
        "thumbnail_studio_enabled",
        "thumbnailStudioEnabled",
    ):
        if bool(us.get(key)):
            return True
    return False


def _thumbnail_hydration_edit_prompt(brief: Dict[str, Any]) -> str:
    """Short edit prompt when the brief already carries fused geo/speech/music/OSD signals."""
    if not isinstance(brief, dict):
        return ""
    pairs: List[str] = []
    for key, label in (
        ("fusion_summary", "Scene intelligence"),
        ("geo_context", "Location / route"),
        ("speech_context", "Speech transcript"),
        ("music_context", "Recognized audio"),
        ("osd_context", "Dashcam HUD"),
        ("trill_context", "Driving energy"),
        ("signal_hashtags", "Evidence tags"),
    ):
        v = str(brief.get(key) or "").strip()
        if v:
            pairs.append(f"{label}: {v[:220]}")
    mass = sum(len(p) for p in pairs)
    if mass < 72:
        return ""
    body = " ".join(pairs)[:900]
    return (
        "Light editorial pass on this existing thumbnail image only. "
        "Preserve overall layout, subject framing, and strict headline / no-text rules from the base. "
        "Subtly tighten realism, colour truth, and environmental cues implied by: "
        f"{body}"
    )[:980]


# ============================================================
# Stage Entry Point
# ============================================================

async def run_thumbnail_stage(ctx: JobContext) -> JobContext:
    """
    Generate candidate thumbnails, score for sharpness, then run AI-powered
    content-category-aware selection to pick the most engaging frame.

    Tier gating:
      - Number of candidates exposed = ctx.entitlements.max_thumbnails
      - AI selection runs only when: OPENAI_API_KEY is set AND
        ctx.entitlements.can_ai is True
      - When AI is not available: falls back cleanly to sharpest frame

    When AI IS available, internally extracts max(max_thumbnails, 4) frames
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

    us = ctx.user_settings or {}
    render_pipeline_pref = normalize_thumbnail_render_pipeline(us)
    auto_thumbnails_on = bool(us.get("auto_thumbnails") or us.get("autoThumbnails"))
    upload_forced_thumbnail_render = _upload_requests_thumbnail_render(us)
    if not auto_thumbnails_on and upload_forced_thumbnail_render:
        logger.info(
            "Thumbnail stage: auto_thumbnails disabled but upload-level studio/pikzels override is enabled"
        )
    elif not auto_thumbnails_on:
        logger.info(
            "Thumbnail stage: auto_thumbnails disabled; generating algorithm-selected raw frame only"
        )

    # ── Tier gates ──────────────────────────────────────────────────────────
    max_thumbnails = 1
    if ctx.entitlements:
        max_thumbnails = max(1, int(getattr(ctx.entitlements, "max_thumbnails", 1) or 1))

    designer_on = user_pref_ai_service_enabled(us, "thumbnail_ai", default=True)
    ai_key_present = bool(OPENAI_API_KEY)
    can_ai = ai_key_present and bool(getattr(ctx.entitlements, "can_ai", False) if ctx.entitlements else False)
    can_ai_designer = can_ai and designer_on

    # When PIKZELS_API_KEY is configured the studio pipeline is the default. Users can still
    # opt-out explicitly by setting either flag to False; we only treat the *absence* of the
    # toggle as "enabled". This matches user expectation that "thumbnails should generate" once
    # the platform integration is wired up — the previous gate required two undocumented
    # user-pref toggles, which is why the Pikzels render path was never reached.
    _renderer_ready_early = studio_renderer_enabled()

    def _opt_in_default_true(*keys: str) -> bool:
        for k in keys:
            if k in us:
                return bool(us.get(k))
        return _renderer_ready_early

    studio_flow_early = _opt_in_default_true("thumbnail_studio_enabled", "thumbnailStudioEnabled")
    use_studio_engine_early = _opt_in_default_true(
        "thumbnail_studio_engine_enabled",
        "thumbnailStudioEngineEnabled",
        "thumbnail_pikzels_enabled",
        "thumbnailPikzelsEnabled",
    )
    pikzels_studio_pipeline = bool(
        designer_on
        and studio_flow_early
        and use_studio_engine_early
        and _renderer_ready_early
    )
    can_custom_thumbs = bool(
        getattr(ctx.entitlements, "can_custom_thumbnails", False) if ctx.entitlements else False
    )
    styled_enabled_early = us.get("styled_thumbnails", us.get("styledThumbnails", True))
    pikzels_overrides_frame_selection = bool(
        pikzels_studio_pipeline
        and can_custom_thumbs
        and styled_enabled_early
        and render_pipeline_pref != "none"
    )

    # When thumbnail designer (AI) is on, extract at least 4 frames for selection.
    extraction_count = max(max_thumbnails, 4 if can_ai_designer else 1)

    # User-specified manual offset (single-thumbnail mode only)
    raw_offset = us.get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        user_offset = float(raw_offset)
        user_offset = max(0.0, min(user_offset, MAX_THUMBNAIL_OFFSET))
    except (TypeError, ValueError):
        user_offset = DEFAULT_THUMBNAIL_OFFSET

    # ── Category detection ──────────────────────────────────────────────────
    hp_cat = getattr(ctx, "hydration_payload", None)
    if isinstance(hp_cat, dict) and str(hp_cat.get("category") or "").strip():
        category = str(hp_cat["category"]).strip().lower()
        category_source = str(hp_cat.get("category_source") or "payload")
    else:
        category = _detect_category(ctx)
        category_source = "thumbnail_detector"

    logger.info(
        "Thumbnail stage: category=%r source=%s upload_id=%s",
        category,
        category_source,
        str(getattr(ctx, "upload_id", "") or ""),
    )

    trace_append(
        ctx,
        "thumbnail_category_detected",
        {
            "category": category,
            "category_source": category_source,
            "hydration_payload": bool(hp_cat),
        },
    )

    logger.info(
        f"Thumbnail stage: video={video_path.name}, "
        f"max_thumbnails={max_thumbnails}, extraction_count={extraction_count}, "
        f"category={category}, "
        f"ai_designer={'on' if can_ai_designer else 'off'} "
        f"(plan={'ok' if can_ai else ('no-key' if not ai_key_present else 'gate')})"
    )

    # ── Duration probe ──────────────────────────────────────────────────────
    duration = await _get_video_duration(video_path)
    logger.debug(f"Video duration: {duration:.1f}s")

    raw_interval = us.get("thumbnail_interval", us.get("thumbnailInterval"))
    min_gap: Optional[float] = None
    if raw_interval is not None:
        try:
            g = float(raw_interval)
            if g > 0:
                min_gap = g
        except (TypeError, ValueError):
            min_gap = None

    # ── Distribute offsets ──────────────────────────────────────────────────
    offsets = _distribute_offsets(
        duration=duration,
        n=extraction_count,
        user_offset=user_offset if extraction_count == 1 else None,
        min_gap_seconds=min_gap,
    )

    # ── VI-derived keyframe offset (Phase #7) ───────────────────────────────
    # When Video Intelligence captured object/logo/person tracks, the most
    # interesting moment in the clip is almost always the midpoint of the
    # highest-confidence non-generic track. We pre-pend that offset so the
    # downstream sharpness + AI-pick pass evaluates it against the evenly
    # distributed candidates. Often it wins outright; if not, the styled
    # render pipeline still gets a high-quality option to feed Pikzels.
    vi_keyframe_offset: Optional[float] = None
    try:
        from services.recognition_engine import select_thumbnail_keyframe_offset
        vi_payload = (
            getattr(ctx, "video_intelligence", None)
            or getattr(ctx, "video_intelligence_context", None)
            or {}
        )
        if isinstance(vi_payload, dict) and not vi_payload.get("error"):
            vi_keyframe_offset = select_thumbnail_keyframe_offset(
                vi_payload, duration_seconds=duration
            )
    except Exception as vi_e:
        logger.debug("Thumbnail: VI keyframe selection failed (non-fatal): %s", vi_e)
    if vi_keyframe_offset is not None and vi_keyframe_offset > 0:
        existing = [round(o, 2) for o in offsets]
        if round(vi_keyframe_offset, 2) not in existing:
            offsets = [vi_keyframe_offset] + offsets
            logger.info(
                "Thumbnail: prepending VI keyframe at %.2fs (object/logo/person peak)",
                vi_keyframe_offset,
            )

    logger.debug(f"Thumbnail offsets: {[f'{o:.1f}s' for o in offsets]}")

    # ── Extract and score all frames ────────────────────────────────────────
    candidates: List[Tuple[Path, float]] = []
    vi_candidate_path: Optional[Path] = None

    for idx, offset in enumerate(offsets):
        out_path = ctx.temp_dir / f"thumb_{ctx.upload_id}_{idx:02d}.jpg"
        success = await _extract_frame(video_path, out_path, offset)

        if not success and offset > 0:
            logger.debug(f"Frame at {offset:.1f}s failed — retrying at t=0")
            success = await _extract_frame(video_path, out_path, 0.0)

        if success:
            score = await _score_sharpness(out_path)
            if score == 0.0:
                score = out_path.stat().st_size / 1_000_000
            # VI keyframe gets a small bonus so the deterministic best moment
            # only loses to a *much* sharper alternative.
            if vi_keyframe_offset is not None and abs(offset - vi_keyframe_offset) < 0.05:
                vi_candidate_path = out_path
                score = score * 1.15 + 0.05
            candidates.append((out_path, score))
            logger.debug(
                f"  Frame {idx}: {out_path.name} @ {offset:.1f}s — "
                f"sharpness={score:.4f}, size={out_path.stat().st_size // 1024}KB"
            )
        else:
            logger.warning(f"  Frame {idx} @ {offset:.1f}s failed — skipping")

    if not candidates:
        logger.warning("Thumbnail generation failed for all offsets (non-fatal)")
        raise SkipStage("FFmpeg thumbnail extraction produced no output")

    # ── Sharpness-best (always computed; used as fallback) ──────────────────
    sharpness_best_path, sharpness_best_score = max(candidates, key=lambda x: x[1])

    # ── AI selection pass ───────────────────────────────────────────────────
    ai_selected_path: Optional[Path] = None
    selection_method = "sharpness"
    selection_mode = normalize_thumbnail_selection_mode(us)
    # Do not skip when pipeline is "ai_edit" — that path runs GPT image edit first,
    # which still benefits from GPT frame selection before any Pikzels step.
    pipeline_studio_first = render_pipeline_pref in ("auto", "studio_renderer") or (
        render_pipeline_pref == "template" and pikzels_studio_pipeline
    )
    skip_ai_frame_pick = bool(
        pikzels_overrides_frame_selection and selection_mode == "ai" and pipeline_studio_first
    )
    if skip_ai_frame_pick:
        logger.info(
            "Thumbnail frame selection: Pikzels/studio pipeline active — "
            "using sharpest frame as compositor input (skipping GPT frame pick)"
        )
        selection_method = "sharpness_pikzels_input"

    if selection_mode == "sharpness" or skip_ai_frame_pick:
        if selection_mode == "sharpness" and not skip_ai_frame_pick:
            logger.debug(
                "Thumbnail frame selection: user pref thumbnailSelectionMode=sharpness — AI pick skipped"
            )
    elif can_ai_designer and len(candidates) > 1:
        try:
            ai_selected_path = await _ai_select_best_frame(candidates, category, ctx)
        except Exception as e:
            logger.warning(f"AI thumbnail selection raised unexpectedly: {e}")
            ai_selected_path = None

        if ai_selected_path and Path(ai_selected_path).exists():
            selection_method = f"ai_{category}"
        else:
            logger.info(
                f"AI selection returned nothing — falling back to sharpest: "
                f"{sharpness_best_path.name}"
            )
    else:
        if selection_mode != "sharpness" and not designer_on:
            reason = "thumbnail designer disabled (pref or aiServiceThumbnailDesigner)"
        elif selection_mode != "sharpness" and not ai_key_present:
            reason = "no API key"
        elif selection_mode != "sharpness" and not can_ai:
            reason = "plan gate"
        elif len(candidates) <= 1:
            reason = "only 1 candidate"
        else:
            reason = "pref sharpness"
        logger.debug(f"AI thumbnail selection skipped: {reason}")

    # ── Final best frame ────────────────────────────────────────────────────
    best_path  = ai_selected_path if ai_selected_path else sharpness_best_path
    best_score = next(
        (s for p, s in candidates if str(p) == str(best_path)),
        sharpness_best_score,
    )

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
    ctx.thumbnail_category = category

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
    if render_pipeline_pref == "none":
        ctx.output_artifacts["thumbnail_render_method"] = "none"
    raw_frame_only = True

    # ── Styled thumbnails (MrBeast-style composite) — Trill + non-Trill, every upload ──
    # Gated by: can_custom_thumbnails + user pref styled_thumbnails (default True)
    can_custom = bool(getattr(ctx.entitlements, "can_custom_thumbnails", False) if ctx.entitlements else False)
    can_ai_style = bool(getattr(ctx.entitlements, "can_ai_thumbnail_styling", False) if ctx.entitlements else False)
    styled_enabled = us.get("styled_thumbnails", us.get("styledThumbnails", True))

    # Diagnostic: log every gate decision so we can see why Pikzels did/didn't run.
    studio_render_report: Dict[str, Any] = {
        "pikzels_api_key_configured": studio_renderer_enabled(),
        "can_custom_thumbnails": can_custom,
        "can_ai_thumbnail_styling": can_ai_style,
        "styled_thumbnails_pref": bool(styled_enabled),
        "render_pipeline_pref": render_pipeline_pref,
        "auto_thumbnails_pref": bool(us.get("auto_thumbnails") or us.get("autoThumbnails")),
        "studio_eligible": False,
        "persona_kind": None,
        "persona_uuid": None,
        "render_steps": [],
        "platform_render_methods": {},
        "hydration_pikzels_edit": {},
        "evidence_anchor": None,
        "skip_reason": None,
        "raw_frame_only": True,
    }
    styled_generation_enabled = bool(can_ai_designer or pikzels_studio_pipeline)
    if not (
        can_custom
        and styled_enabled
        and ctx.temp_dir
        and render_pipeline_pref != "none"
        and styled_generation_enabled
    ):
        if not can_custom:
            studio_render_report["skip_reason"] = "tier lacks can_custom_thumbnails"
        elif not styled_enabled:
            studio_render_report["skip_reason"] = "user pref styled_thumbnails=false"
        elif not ctx.temp_dir:
            studio_render_report["skip_reason"] = "no temp dir"
        elif render_pipeline_pref == "none":
            studio_render_report["skip_reason"] = "thumbnailRenderPipeline=none"
        elif not styled_generation_enabled:
            studio_render_report["skip_reason"] = "ai/pikzels disabled -> raw frame only"
        ctx.output_artifacts["studio_render_report"] = json.dumps(studio_render_report)
        try:
            from services.diag_persist import schedule_persist_artifact_now

            schedule_persist_artifact_now(ctx, "studio_render_report")
        except Exception:
            pass
        logger.info(
            "[thumb-renderer] styled-thumbnail block skipped reason=%s upload=%s",
            studio_render_report["skip_reason"],
            ctx.upload_id,
        )

    if (
        can_custom
        and styled_enabled
        and ctx.temp_dir
        and render_pipeline_pref != "none"
        and styled_generation_enabled
    ):
        brief: Optional[Dict] = None
        if can_ai_designer:
            brief = await _generate_thumbnail_brief(ctx, category)
        if not brief and can_ai_designer:
            brief = _sanitize_thumbnail_brief(ctx, None, category, note="")
        elif not brief:
            raw_brief: Dict[str, Any] = {}
            if pikzels_studio_eligible_for_styled_thumbnail(us, ctx.entitlements, require_auto_thumbnails=False):
                text_brief = await generate_pikzels_text_brief(
                    source_title=ctx.get_effective_title() or ctx.filename or "UploadM8 video",
                    niche=category,
                    context_summary=ctx.get_thumbnail_brief_vars(category=category).get("fusion_summary", ""),
                )
                if text_brief:
                    raw_brief["pikzels_text_brief"] = text_brief
            # Do not set ``notes`` to a placeholder here — it blocked evidence hydration
            # (anchor phrase) from ever reaching ``notes`` and then the Pikzels prompt.
            brief = _sanitize_thumbnail_brief(ctx, raw_brief, category, note="")
        else:
            brief = _sanitize_thumbnail_brief(ctx, brief, category)

        trace_append(
            ctx,
            "thumbnail_brief_pre_strategy",
            {
                "can_ai_designer": bool(can_ai_designer),
                "selected_headline": str((brief or {}).get("selected_headline") or "")[:80],
                "has_fusion_summary": bool(str((brief or {}).get("fusion_summary") or "").strip()),
                "has_hydration_story": bool(str((brief or {}).get("hydration_story") or "").strip()),
                "has_pikzels_text_brief": bool(str((brief or {}).get("pikzels_text_brief") or "").strip()),
            },
        )

        brief = _apply_thumbnail_default_strategy(brief, us, category=category)

        # Hydrate the brief with deterministic evidence pulled from every analyzed
        # signal on ctx (vision labels, OSD speed, geo, ACR music, Whisper).
        # When a field is already set on the brief we keep it; we only fill gaps.
        # Without this, the brief depends entirely on the GPT brief writer; if
        # GPT was off or skipped, Pikzels rendered with a thin brief and the
        # output looked just as generic as the model captions.
        try:
            from services.hydration_enforcer import (
                build_anchor_phrase,
                build_evidence_hashtags,
                collect_evidence,
            )

            brief = apply_hydration_payload_to_thumbnail_brief(ctx, brief)
            hp_ctx = getattr(ctx, "hydration_payload", None)
            if isinstance(hp_ctx, dict) and str(hp_ctx.get("anchor_phrase") or "").strip():
                studio_render_report["evidence_anchor"] = str(hp_ctx["anchor_phrase"])[:500]
            else:
                pool = collect_evidence(ctx)
                anchor = build_anchor_phrase(pool, ctx)
                studio_render_report["evidence_anchor"] = anchor or None
                if anchor:
                    _ph_notes = frozenset({"No AI — evidence-based brief", "Fallback brief"})
                    _cur_notes = str(brief.get("notes") or "").strip()
                    if not _cur_notes or _cur_notes in _ph_notes:
                        brief["notes"] = anchor[:200]
                    if not str(brief.get("speech_context") or "").strip() and pool.transcript_phrase:
                        brief["speech_context"] = pool.transcript_phrase[:260]
                    if not str(brief.get("music_context") or "").strip() and (
                        pool.music_artist or pool.music_title
                    ):
                        parts = [p for p in (pool.music_artist, pool.music_title) if p]
                        brief["music_context"] = " — ".join(parts)[:220]
                    if not str(brief.get("geo_context") or "").strip():
                        geo_bits_fb: List[str] = []
                        if pool.road:
                            geo_bits_fb.append(pool.road)
                        if pool.gazetteer_place and pool.state_abbr:
                            geo_bits_fb.append(f"{pool.gazetteer_place}, {pool.state_abbr}")
                        elif pool.city and pool.state_abbr:
                            geo_bits_fb.append(f"{pool.city}, {pool.state_abbr}")
                        elif pool.protected_area:
                            geo_bits_fb.append(pool.protected_area)
                        if geo_bits_fb:
                            brief["geo_context"] = "; ".join(geo_bits_fb)[:260]
                    if not str(brief.get("osd_context") or "").strip() and (
                        pool.max_speed_mph or pool.driver_name
                    ):
                        osd_bits_fb: List[str] = []
                        if pool.max_speed_mph:
                            osd_bits_fb.append(f"max {int(round(pool.max_speed_mph))} MPH")
                        if pool.driver_name:
                            osd_bits_fb.append(f"driver {pool.driver_name}")
                        brief["osd_context"] = "; ".join(osd_bits_fb)[:220]
                    if not str(brief.get("trill_context") or "").strip() and pool.trill_bucket:
                        brief["trill_context"] = (
                            f"Trill bucket {pool.trill_bucket} (score {pool.trill_score:.0f})"
                        )[:220]
                    evidence_tags = build_evidence_hashtags(pool, max_extra=8)
                    if evidence_tags and not str(brief.get("signal_hashtags") or "").strip():
                        brief["signal_hashtags"] = ", ".join(evidence_tags[:8])[:180]
        except Exception as e:
            logger.debug("[thumb-renderer] hydration brief enrichment failed: %s", e)

        trace_append(
            ctx,
            "thumbnail_brief_post_evidence",
            {
                "anchor": (studio_render_report.get("evidence_anchor") or "")[:120],
                "selected_headline": str((brief or {}).get("selected_headline") or "")[:80],
                "fusion_chars": len(str((brief or {}).get("fusion_summary") or "")),
                "hydration_story_chars": len(str((brief or {}).get("hydration_story") or "")),
            },
        )

        ctx.output_artifacts["thumbnail_brief_json"] = json.dumps(brief)
        try:
            from services.diag_persist import schedule_persist_artifact_now

            schedule_persist_artifact_now(ctx, "thumbnail_brief_json")
        except Exception:
            pass

        # TikTok publishing may still use a frame offset depending on API support, but
        # UploadM8 should render/store a generated TikTok cover preview like every
        # other selected platform so queue/detail UI can display a complete set.
        platform_map: Dict[str, str] = {}
        tiktok_plan = brief.get("platform_plan", {}).get("tiktok", {})
        ctx.output_artifacts["tiktok_thumb_offset_seconds"] = str(
            tiktok_plan.get("thumb_offset_seconds", 1.5)
        )

        # Render per selected platform (YouTube 16:9, Instagram/Facebook/TikTok 9:16).
        _plat_lower = [str(pl).strip().lower() for pl in (ctx.platforms or []) if str(pl).strip()]
        platforms_to_render = [
            p
            for p in ("youtube", "instagram", "facebook", "tiktok")
            if (brief.get("platform_plan", {}).get(p, {}).get("enabled", True)) and p in _plat_lower
        ]
        render_method = "none"
        primary_styled: Optional[Path] = None  # Prefer YouTube for primary
        persona_api, studio_opts = _studio_persona_for_request(us)
        if isinstance(persona_api, dict):
            studio_render_report["persona_kind"] = persona_api.get("kind")
            studio_render_report["persona_uuid"] = persona_api.get("id")

        studio_ok = bool(
            pikzels_studio_eligible_for_styled_thumbnail(
                us, ctx.entitlements, require_auto_thumbnails=False
            )
            and isinstance(brief, dict)
        )
        ai_edit_ok = bool(can_ai_designer and can_ai_style and OPENAI_API_KEY)
        strict_studio = _strict_studio_mode_enabled(us)
        if strict_studio and not studio_ok:
            raise ThumbnailError(
                "Strict Pikzels mode enabled but Studio renderer is unavailable for this upload"
            )
        # Legacy "template only" pipeline would skip Pikzels entirely; when the studio
        # engine is enabled, prefer auto ordering so the API render runs first.
        effective_render_pipeline = render_pipeline_pref
        if studio_ok and render_pipeline_pref == "template":
            effective_render_pipeline = "auto"
            logger.info(
                "thumbnail_render_pipeline=template overridden to auto "
                "(Pikzels/studio renderer active for this job)"
            )
        render_steps = _thumbnail_styled_render_order(
            effective_render_pipeline,
            studio_ok=studio_ok,
            ai_edit_ok=ai_edit_ok,
        )
        studio_render_report["studio_eligible"] = studio_ok
        studio_render_report["render_steps"] = list(render_steps)

        trace_append(
            ctx,
            "thumbnail_render_pipeline",
            {
                "platforms": list(platforms_to_render),
                "studio_eligible": bool(studio_ok),
                "render_steps": list(render_steps),
                "effective_render_pipeline": effective_render_pipeline,
            },
        )

        if not platforms_to_render:
            platforms_to_render = ["youtube"]
            logger.info(
                "Styled thumbnails: no recognized upload targets — rendering default 16:9 cover"
            )

        for platform in platforms_to_render:
            out_name = f"thumb_styled_{platform}_{ctx.upload_id}.jpg"
            out_path = ctx.temp_dir / out_name
            ok = False
            attempted_methods: List[str] = []
            for step in render_steps:
                if step == "studio" and studio_ok:
                    attempted_methods.append("studio_renderer")
                    trace_append(ctx, "thumbnail_pikzels_attempt", {"platform": platform})
                    ok = await render_thumbnail_with_studio_renderer(
                        best_path,
                        brief,
                        platform,
                        out_path,
                        upload_id=str(ctx.upload_id),
                        category=category,
                        persona=persona_api,
                        options=studio_opts,
                        job_context=ctx,
                    )
                    if ok:
                        render_method = "studio_renderer"
                        hp = _thumbnail_hydration_edit_prompt(brief or {})
                        if hp and _hydration_pikzels_edit_enabled():
                            he_ok = await refine_thumbnail_with_pikzels_edit(
                                out_path,
                                hp,
                                platform=platform,
                                upload_id=str(ctx.upload_id),
                            )
                            studio_render_report["hydration_pikzels_edit"][platform] = bool(
                                he_ok
                            )
                    else:
                        logger.warning(
                            "[thumb-renderer] Pikzels studio render failed for %s upload=%s — "
                            "falling back to next render step (was: %s)",
                            platform, ctx.upload_id, render_steps,
                        )
                elif step == "ai_edit" and ai_edit_ok:
                    attempted_methods.append("ai_edit")
                    ok = await _ai_edit_thumbnail(best_path, brief, out_path, retry_reduce=False)
                    if not ok:
                        ok = await _ai_edit_thumbnail(best_path, brief, out_path, retry_reduce=True)
                    if ok:
                        render_method = "ai_edit"
                elif step == "template":
                    attempted_methods.append("template")
                    ok = _render_template_thumbnail(best_path, brief, platform, out_path)
                    if ok:
                        render_method = "template"
                if ok:
                    break
            studio_render_report["platform_render_methods"][platform] = {
                "attempted": attempted_methods,
                "succeeded_with": render_method if ok else None,
            }
            if ok:
                platform_map[platform] = str(out_path)
                if primary_styled is None or platform == "youtube":
                    primary_styled = out_path
        if primary_styled and primary_styled.exists():
            ctx.thumbnail_path = primary_styled
            ctx.output_artifacts["thumbnail"] = str(primary_styled)
            raw_frame_only = False
            studio_render_report["raw_frame_only"] = False
        elif strict_studio:
            raise ThumbnailError(
                "Strict Pikzels mode enabled and Studio renderer did not produce a styled thumbnail"
            )

        ctx.output_artifacts["thumbnail_render_method"] = render_method
        ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(platform_map)
        ctx.output_artifacts["studio_render_report"] = json.dumps(studio_render_report)
        try:
            from services.diag_persist import schedule_persist_artifact_now

            schedule_persist_artifact_now(
                ctx,
                "studio_render_report",
                "thumbnail_render_method",
                "platform_thumbnail_map",
            )
        except Exception:
            pass
        logger.info(
            "[thumb-renderer] studio render report upload=%s eligible=%s steps=%s "
            "persona=%s methods=%s anchor=%r",
            ctx.upload_id,
            studio_render_report["studio_eligible"],
            studio_render_report["render_steps"],
            studio_render_report.get("persona_kind"),
            studio_render_report["platform_render_methods"],
            studio_render_report.get("evidence_anchor"),
        )

        trace_append(
            ctx,
            "thumbnail_styled_block_complete",
            {
                "thumbnail_render_method": render_method,
                "platform_map_keys": list(platform_map.keys()),
                "raw_frame_only": bool(raw_frame_only),
            },
        )

    if raw_frame_only:
        trace_append(
            ctx,
            "thumbnail_raw_frame_only",
            {
                "reason": str(
                    (studio_render_report.get("skip_reason") if isinstance(studio_render_report, dict) else "")
                    or "styled render not produced"
                )[:180],
            },
        )
    ctx.output_artifacts["thumbnail_raw_frame_only"] = bool(raw_frame_only)

    trace_append(
        ctx,
        "thumbnail_stage_complete",
        {
            "selection_method": selection_method,
            "frames": len(candidates),
            "best_frame": getattr(best_path, "name", str(best_path)),
        },
    )

    logger.info(
        f"Thumbnail stage complete: {len(candidates)} frames, "
        f"selected={best_path.name} "
        f"(method={selection_method}, sharpness={best_score:.4f})"
    )

    return ctx
