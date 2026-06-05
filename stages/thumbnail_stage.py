"""
UploadM8 Thumbnail Stage — Pikzels-first
========================================
Sample a few JPEG frames from the source video (via ``imageio`` / ``imageio-ffmpeg``,
not PATH ``ffmpeg``/``ffprobe``), score them with a local sharpness heuristic, pick
the sharpest, then run the **Pikzels v2** styled pipeline when configured.

Flow:
  1. Probe video duration (imageio reader metadata)
  2. Distribute N frame offsets (N capped; no GPT frame pick)
  3. Decode each frame to a 1080px-wide JPEG
  4. Score sharpness (gradient variance — no FFmpeg blurdetect)
  5. Detect content category (user hint → filename → general)
  6. Pick sharpest candidate (GPT vision selection removed)
  7. Set ctx.thumbnail_path / ctx.thumbnail_paths / ctx.thumbnail_scores
  8. Store metadata in ctx.output_artifacts
  9. When Pikzels studio is enabled: brief from **Pikzels text brief** + hydration
     (no GPT JSON brief), then render per platform via **Pikzels v2 image** only;
     optional Pikzels ``/v2/thumbnail/edit`` pass when hydration edit is enabled.

Fallback chain:
  - Extraction fails at an offset: retry near t≈0
  - Everything fails: raise SkipStage (non-fatal — pipeline continues)

NOTE: R2 upload and DB persistence are handled by worker.py AFTER this stage.

Exports: run_thumbnail_stage(ctx)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.content_attribution import (
    normalize_thumbnail_render_pipeline,
    normalize_thumbnail_selection_mode,
)
from core.thumbnail_text import (
    CATEGORY_HEADLINE_FALLBACKS,
    clean_thumbnail_headline,
    is_generic_thumbnail_headline,
)

from .context import JobContext
from .errors import SkipStage, ThumbnailError
from services.thumbnail_brief_pipeline import (
    attach_youtube_support_image_from_ctx,
    copy_brief_for_persistence,
    finalize_styled_thumbnail_brief,
)
from services.thumbnail_studio import closeness_to_pikzels_image_weight
from services.thumbnail_trace import trace_append
from services.thumbnail_sticker_pack import build_sticker_pack, sticker_pack_to_json
from services.thumbnail_sticker_render import render_sticker_composite, sticker_composite_enabled

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


def _dashcam_pov_content(ctx: JobContext, category: str) -> bool:
    """POV dashcam clip with no expressive on-screen face — Pikzels must not invent people."""
    cat = (category or "").strip().lower()
    if cat == "dashcam":
        return True
    fname = (getattr(ctx, "filename", "") or "").upper()
    if any(tok in fname for tok in ("CAM_", "DASHCAM", "_EVNT", "BLACKVU", "THINKWARE")):
        return True
    vc = ctx.vision_context or {}
    if isinstance(vc, dict):
        labels = " ".join(str(x).lower() for x in (vc.get("label_names") or []))
        dashcam_markers = (
            "windshield",
            "windscreen",
            "rear-view mirror",
            "automotive mirror",
            "automotive exterior",
            "hood",
            "dashcam",
        )
        if any(m in labels for m in dashcam_markers):
            try:
                faces = int(vc.get("face_count") or 0)
            except (TypeError, ValueError):
                faces = 0
            if faces == 0 and not bool(vc.get("expressive")):
                return True
    return False


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
    dashcam_pov = _dashcam_pov_content(ctx, category)
    if dashcam_pov:
        out["directional_element"] = "none"
        out["emotion_cue"] = ""
        out["color_mood"] = "blue_white"
        out["props"] = []
        out["_uploadm8_dashcam_pov"] = True
    else:
        if out.get("directional_element") not in {"arrow_up", "arrow_right", "circle", "glow_box", "none"}:
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
# Template Renderer (PIL overlays — deterministic fallback)
# ============================================================

def _render_template_thumbnail(
    base_path: Path,
    brief: Dict,
    platform: str,
    output_path: Path,
    *,
    platform_color: Optional[str] = None,
    accent_color: Optional[str] = None,
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

    # Badge colors — user platform color wins when configured
    badge_style = brief.get("badge_style", "red")
    badge_colors = {"red": "#e53935", "yellow": "#fdd835", "white": "#ffffff", "black": "#1a1a1a"}
    if platform_color:
        from services.platform_colors import contrasting_text_color

        badge_bg = platform_color
        badge_fg = contrasting_text_color(platform_color)
    else:
        badge_fg = "#ffffff" if badge_style in ("red", "black") else "#1a1a1a"
        badge_bg = badge_colors.get(badge_style, "#e53935")
    highlight = accent_color or "#FFFFFF"

    # Draw badge (top-left)
    if badge_text:
        pad = 8
        bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
        bw, bh = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
        draw.rectangle([safe_margin, safe_margin, safe_margin + bw, safe_margin + bh], fill=badge_bg, outline=highlight, width=2)
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
        draw.ellipse([cx-50, cy-50, cx+50, cy+50], outline=highlight, width=4)
    elif elem in ("arrow_up", "arrow_right"):
        cx, cy = target_w - safe_margin - 60, target_h - safe_margin - 60
        draw.polygon([(cx, cy-30), (cx-20, cy+20), (cx+20, cy+20)], outline=highlight, width=3)

    try:
        img.save(output_path, "JPEG", quality=90)
        return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE
    except Exception as e:
        logger.warning(f"Template render save failed: {e}")
        return False



async def _get_video_duration(video_path: Path) -> float:
    from services import thumbnail_frame_extract as tfe

    return float(await asyncio.to_thread(tfe.video_duration_seconds, video_path))


async def _extract_frame(video_path: Path, output_path: Path, offset: float) -> bool:
    from services import thumbnail_frame_extract as tfe

    return bool(
        await asyncio.to_thread(tfe.extract_jpeg_at_offset, video_path, output_path, float(offset))
    )


async def _score_sharpness(image_path: Path) -> float:
    """Laplacian-gradient variance (higher = sharper). Normalized to ~0–1 scale."""
    from services import thumbnail_frame_extract as tfe

    raw = float(await asyncio.to_thread(tfe.laplacian_variance_score, image_path))
    if raw <= 0:
        return 0.0
    # Typical JPEG scores are single digits to low hundreds; squash for comparability
    return max(0.0, min(1.0, raw / 200.0))


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
    sticker_ok: bool = False,
) -> List[str]:
    """
    Per-platform compositing attempts. Values: sticker | studio | ai_edit | template.

    When ``sticker_ok`` (``THUMBNAIL_STICKER_COMPOSITE=1``), local PIL sticker
    compositing runs first using real frame crops — no generative API spend. Pikzels
    studio still runs next when configured so YouTube reference styling can apply
    if the local pass did not produce an acceptable cover.

    When Pikzels studio is available (``studio_ok``), styled thumbnail pixels prefer
    that path after stickers. Optional ``refine_thumbnail_with_pikzels_edit`` still
    runs after a successful studio render when hydration edit is enabled.
    """
    p = (pipeline_pref or "auto").lower().strip()
    if p not in ("auto", "studio_renderer", "ai_edit", "template", "none"):
        p = "auto"
    if p == "none":
        return []

    sticker_first: List[str] = ["sticker"] if sticker_ok else []

    if studio_ok:
        return sticker_first + ["studio"]
    if p == "template":
        return sticker_first + ["template"]
    if p == "ai_edit":
        order: List[str] = list(sticker_first)
        if ai_edit_ok:
            order.append("ai_edit")
        order.append("template")
        return order
    out: List[str] = list(sticker_first)
    if ai_edit_ok:
        out.append("ai_edit")
    out.append("template")
    return out


_CANONICAL_STUDIO_PLATFORMS: tuple[str, ...] = ("youtube", "instagram", "facebook", "tiktok")


def styled_thumbnail_platform_targets(
    selected_platforms: Any,
    *,
    platform_plan: Any,
) -> List[str]:
    """Platforms to run studio (Pikzels) / template renders for.

    When the upload has **no** ``platforms`` list (legacy rows or ingestion gaps),
    returns every **enabled** entry from ``platform_plan`` in canonical order so we
    still generate 16:9 and 9:16 covers for publish + UI. When ``platforms`` is
    non-empty, only those targets are rendered.
    """
    plan = platform_plan if isinstance(platform_plan, dict) else {}

    def _plan_entry_enabled(p: str) -> bool:
        entry = plan.get(p)
        if isinstance(entry, dict):
            return bool(entry.get("enabled", True))
        return True

    plat_lower = [str(pl).strip().lower() for pl in (selected_platforms or []) if str(pl).strip()]
    ordered = [p for p in _CANONICAL_STUDIO_PLATFORMS if _plan_entry_enabled(p)]
    if plat_lower:
        return [p for p in ordered if p in plat_lower]
    return list(ordered)


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

    Policy (forced-Pikzels mode): when ``PIKZELS_API_KEY`` is configured and the
    user has not explicitly opted out, run Pikzels for every selected platform.
    Stale rows that left ``thumbnail_render_pipeline = 'none'`` are coerced to
    ``auto`` and the ``thumbnail_studio_*`` flags default to True so first-time
    users with the API key on the server see studio renders.

    Auto-thumbnails off normally skips the thumbnail stage's scheduled work, but when
    the Pikzels API key is configured and the tier supports custom thumbnails we still
    run studio renders (unless the upload explicitly opts out via studio prefs).
    """
    ready = studio_renderer_enabled()
    can_custom = bool(getattr(entitlements, "can_custom_thumbnails", False) if entitlements else False)
    can_ai_style = bool(getattr(entitlements, "can_ai_thumbnail_styling", False) if entitlements else False)

    # Creator Lite and below: local template renderer only (no Pikzels API — avoids 402 noise).
    if ready and can_custom and not can_ai_style:
        return False

    auto_on = bool(us.get("auto_thumbnails") or us.get("autoThumbnails"))
    if require_auto_thumbnails and not auto_on:
        if not _upload_requests_thumbnail_render(us) and not (ready and can_custom):
            return False

    render_pipeline_pref = normalize_thumbnail_render_pipeline(us)
    if render_pipeline_pref == "none":
        # When the API key is configured AND the user has the entitlement, treat
        # a stale "none" preference as opt-in (auto). Without this every account
        # whose preference row predates the studio renderer silently bypassed Pikzels.
        if ready and can_custom:
            render_pipeline_pref = "auto"
        else:
            return False

    explicit_styled = (
        us.get("styled_thumbnails") if "styled_thumbnails" in us
        else (us.get("styledThumbnails") if "styledThumbnails" in us else None)
    )
    if explicit_styled is None:
        styled_enabled = True if (ready and can_custom) else True
    else:
        styled_enabled = bool(explicit_styled)
    if not (can_custom and styled_enabled):
        return False

    def _opt_in_default_true(*keys: str) -> bool:
        for k in keys:
            if k in us:
                return bool(us.get(k))
        return True if ready else False

    studio_flow = _opt_in_default_true("thumbnail_studio_enabled", "thumbnailStudioEnabled")
    use_studio_engine = _opt_in_default_true(
        "thumbnail_studio_engine_enabled",
        "thumbnailStudioEngineEnabled",
        "thumbnail_pikzels_enabled",
        "thumbnailPikzelsEnabled",
    )
    return bool(studio_flow and use_studio_engine and ready)


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


def effective_thumbnail_category(us: Dict[str, Any], detected_category: str) -> str:
    """Prefer saved Studio default audience/niche over auto-detected category."""
    strategy = _thumbnail_default_strategy(us)
    niche = str(strategy.get("audience_niche") or "").strip()
    if niche:
        from services.thumbnail_niches import normalize_niche

        return normalize_niche(niche, default=detected_category or "general")
    return (detected_category or "general").strip().lower() or "general"


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
    Sample frames, pick the sharpest, optionally run the Pikzels styled renderer.

    Tier gating:
      - Number of candidate frames exposed = ctx.entitlements.max_thumbnails
      - GPT vision frame selection and GPT JSON thumbnail briefs are disabled.

    ctx fields set by this stage:
      ctx.thumbnail_path        — sharpest extracted frame (Pikzels compositor input)
      ctx.thumbnail_paths       — candidates up to max_thumbnails
      ctx.thumbnail_scores      — {str(path): score}
      ctx.output_artifacts      — thumbnail metadata keys
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
    can_custom_ent = bool(getattr(ctx.entitlements, "can_custom_thumbnails", False) if ctx.entitlements else False)
    # Keep in sync with ``pikzels_studio_eligible_for_styled_thumbnail``: legacy "none"
    # must not short-circuit the styled block before eligibility coerces → auto.
    if studio_renderer_enabled() and can_custom_ent and render_pipeline_pref == "none":
        render_pipeline_pref = "auto"
        logger.info(
            "[thumb] thumbnailRenderPipeline coerced none→auto (Pikzels key + can_custom_thumbnails) upload=%s",
            getattr(ctx, "upload_id", "") or "",
        )
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

    # Thumbnail pixels come from Pikzels when the engine is configured; GPT/OpenAI
    # is not used for frame pick or brief JSON. ``thumbnail_ai`` pref no longer gates studio.
    pikzels_studio_pipeline_on = pikzels_studio_eligible_for_styled_thumbnail(
        us,
        ctx.entitlements,
        require_auto_thumbnails=False,
    )
    # Product policy: no OpenAI image-edit for thumbnails — Pikzels or PIL template only.
    ai_edit_ok = False

    # Sample several frames; pick sharpest (no GPT vision).
    extraction_count = min(6, max(3, max_thumbnails))

    # ── Category detection (before logging / frame work) ────────────────────
    hp_cat = getattr(ctx, "hydration_payload", None)
    if isinstance(hp_cat, dict) and str(hp_cat.get("category") or "").strip():
        category = str(hp_cat["category"]).strip().lower()
        category_source = str(hp_cat.get("category_source") or "payload")
    else:
        category = _detect_category(ctx)
        category_source = "thumbnail_detector"

    category = effective_thumbnail_category(us, category)
    if category_source == "thumbnail_detector" and _thumbnail_default_strategy(us).get(
        "audience_niche"
    ):
        category_source = "studio_default_strategy"

    logger.info(
        f"Thumbnail stage: video={video_path.name}, "
        f"max_thumbnails={max_thumbnails}, extraction_count={extraction_count}, "
        f"category={category}, "
        f"pikzels_studio_pipeline={'on' if pikzels_studio_pipeline_on else 'off'}, "
        f"openai_image_edit_for_thumbnails=False"
    )
    raw_offset = us.get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        user_offset = float(raw_offset)
        user_offset = max(0.0, min(user_offset, MAX_THUMBNAIL_OFFSET))
    except (TypeError, ValueError):
        user_offset = DEFAULT_THUMBNAIL_OFFSET

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
    candidates: List[Tuple[Path, float, float]] = []
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
            candidates.append((out_path, score, float(offset)))
            logger.debug(
                f"  Frame {idx}: {out_path.name} @ {offset:.1f}s — "
                f"sharpness={score:.4f}, size={out_path.stat().st_size // 1024}KB"
            )
        else:
            logger.warning(f"  Frame {idx} @ {offset:.1f}s failed — skipping")

    if not candidates:
        logger.warning("Thumbnail generation failed for all offsets (non-fatal)")
        raise SkipStage("Thumbnail frame extraction produced no output")

    # ── Sharpness-best (no GPT frame pick) ──────────────────────────────────
    sharpness_best_path, sharpness_best_score, best_frame_offset = max(
        candidates, key=lambda x: x[1]
    )
    selection_method = "sharpness"
    if normalize_thumbnail_selection_mode(us) != "sharpness":
        logger.debug(
            "Thumbnail frame selection: thumbnailSelectionMode=%r ignored — GPT frame pick disabled",
            normalize_thumbnail_selection_mode(us),
        )

    best_path = sharpness_best_path
    best_score = sharpness_best_score

    # ── Populate ctx ────────────────────────────────────────────────────────
    # Chronological candidate list (caption_stage uses these for multi-frame story)
    # Respect max_thumbnails tier cap for the exposed list
    ctx.thumbnail_paths = [p for p, _, _ in candidates[:max_thumbnails]]
    ctx.thumbnail_scores = {str(p): s for p, s, _ in candidates}

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
    ctx.output_artifacts["thumbnail_frame_offset_seconds"] = str(round(best_frame_offset, 3))
    ctx.output_artifacts["thumbnail_candidates"]        = json.dumps(
        [str(p) for p in ctx.thumbnail_paths]
    )
    ctx.output_artifacts["thumbnail_scores"]            = json.dumps(
        {str(p): round(s, 4) for p, s, _ in candidates}
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
    styled_generation_enabled = bool(pikzels_studio_pipeline_on)
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
            studio_render_report["skip_reason"] = "pikzels studio pipeline disabled or API key missing"
        ctx.output_artifacts["studio_render_report"] = json.dumps(studio_render_report)
        try:
            from services.diag_persist import schedule_persist_artifact_now

            schedule_persist_artifact_now(ctx, "studio_render_report")
        except Exception:
            pass
        if studio_render_report.get("pikzels_api_key_configured"):
            logger.warning(
                "[thumb-renderer] PIKZELS_API_KEY set but styled block skipped reason=%s upload=%s",
                studio_render_report["skip_reason"],
                ctx.upload_id,
            )
        else:
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
        raw_brief: Dict[str, Any] = {}
        if pikzels_studio_eligible_for_styled_thumbnail(us, ctx.entitlements, require_auto_thumbnails=False):
            text_brief = await generate_pikzels_text_brief(
                source_title=ctx.get_effective_title() or ctx.filename or "UploadM8 video",
                niche=category,
                context_summary=ctx.get_thumbnail_brief_vars(category=category).get("fusion_summary", ""),
            )
            if text_brief:
                raw_brief["pikzels_text_brief"] = text_brief
        brief = _sanitize_thumbnail_brief(ctx, raw_brief, category, note="")

        trace_append(
            ctx,
            "thumbnail_brief_pre_strategy",
            {
                "brief_source": "pikzels_text_plus_hydration",
                "selected_headline": str((brief or {}).get("selected_headline") or "")[:80],
                "has_fusion_summary": bool(str((brief or {}).get("fusion_summary") or "").strip()),
                "has_hydration_story": bool(str((brief or {}).get("hydration_story") or "").strip()),
                "has_pikzels_text_brief": bool(str((brief or {}).get("pikzels_text_brief") or "").strip()),
            },
        )

        brief = finalize_styled_thumbnail_brief(
            ctx,
            brief,
            category,
            us,
            studio_render_report=studio_render_report,
            evidence_anchor_fallbacks=True,
        )
        brief = attach_youtube_support_image_from_ctx(brief, ctx)

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

        ctx.output_artifacts["thumbnail_brief_json"] = json.dumps(copy_brief_for_persistence(brief))
        ctx.output_artifacts["thumbnail_dashcam_pov"] = (
            "1" if (brief or {}).get("_uploadm8_dashcam_pov") else "0"
        )
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
        platforms_to_render = styled_thumbnail_platform_targets(
            ctx.platforms,
            platform_plan=brief.get("platform_plan") or {},
        )
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
            sticker_ok=sticker_composite_enabled(),
        )
        studio_render_report["studio_eligible"] = studio_ok
        studio_render_report["render_steps"] = list(render_steps)
        studio_render_report["sticker_composite_enabled"] = sticker_composite_enabled()

        frame_offset_s = float(
            ctx.output_artifacts.get("thumbnail_frame_offset_seconds") or best_frame_offset
        )
        sticker_pack = build_sticker_pack(ctx, frame_offset_s) if sticker_composite_enabled() else []
        ctx.output_artifacts["sticker_pack_json"] = sticker_pack_to_json(sticker_pack)
        studio_render_report["sticker_count"] = len(sticker_pack)

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
                "Styled thumbnails: platform_plan disabled all targets — "
                "using YouTube 16:9 fallback cover"
            )

        pikzels_render_failures: List[Dict[str, Any]] = []
        for platform in platforms_to_render:
            out_name = f"thumb_styled_{platform}_{ctx.upload_id}.jpg"
            out_path = ctx.temp_dir / out_name
            ok = False
            attempted_methods: List[str] = []
            studio_attempted = False
            for step in render_steps:
                if step == "sticker" and sticker_composite_enabled():
                    attempted_methods.append("sticker_composite")
                    from services.platform_colors import platform_color_for, resolve_platform_colors

                    _plat_colors = resolve_platform_colors(us)
                    ok = await render_sticker_composite(
                        best_path,
                        brief,
                        platform,
                        out_path,
                        sticker_pack,
                        platform_color=platform_color_for(_plat_colors, platform),
                        accent_color=_plat_colors.get("accent"),
                    )
                    if ok:
                        render_method = "sticker_composite"
                elif step == "studio" and studio_ok:
                    attempted_methods.append("studio_renderer")
                    studio_attempted = True
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
                        skip_hydration_edit = bool((brief or {}).get("_uploadm8_dashcam_pov"))
                        if hp and _hydration_pikzels_edit_enabled() and not skip_hydration_edit:
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
                        # Capture the last pikzels HTTP status from the provider-error trace so
                        # the worker can emit a focused admin ops_incident. The trace lives in
                        # ctx.output_artifacts["provider_error_trace"] as a JSON list.
                        last_status: Any = "unknown"
                        last_message = ""
                        last_code = ""
                        try:
                            raw_trace = ctx.output_artifacts.get("provider_error_trace") if isinstance(ctx.output_artifacts, dict) else None
                            if isinstance(raw_trace, str) and raw_trace.strip():
                                _rows = json.loads(raw_trace)
                                if isinstance(_rows, list):
                                    for _r in reversed(_rows):
                                        if isinstance(_r, dict) and str(_r.get("provider") or "").lower() == "pikzels":
                                            last_status = _r.get("http_status", "unknown") or "unknown"
                                            last_message = str(_r.get("message") or "")[:240]
                                            last_code = str(_r.get("provider_code") or "").strip()
                                            break
                        except Exception:
                            pass
                        _snippet = ""
                        try:
                            _snippet = str(
                                (_r.get("response_body_snippet") if isinstance(_r, dict) else "") or ""
                            )
                        except Exception:
                            _snippet = ""
                        _combined = f"{last_message} {_snippet}".lower()
                        _reason = last_code or (
                            "prompt_too_long"
                            if "prompt" in _combined and ("1200" in _combined or "1000" in _combined)
                            else "pikzels_http_error"
                        )
                        pikzels_render_failures.append({
                            "provider": "pikzels",
                            "status": "failed",
                            "reason": _reason,
                            "fallback": "auto_frame",
                            "platform": platform,
                            "http_status": last_status,
                            "message": last_message,
                        })
                elif step == "template":
                    attempted_methods.append("template")
                    from services.platform_colors import platform_color_for, resolve_platform_colors

                    _plat_colors = resolve_platform_colors(us)
                    ok = _render_template_thumbnail(
                        best_path,
                        brief,
                        platform,
                        out_path,
                        platform_color=platform_color_for(_plat_colors, platform),
                        accent_color=_plat_colors.get("accent"),
                    )
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
        if pikzels_render_failures:
            ctx.output_artifacts["pikzels_render_failures"] = json.dumps(pikzels_render_failures)
        try:
            from services.diag_persist import schedule_persist_artifact_now

            persist_keys = ["studio_render_report", "thumbnail_render_method", "platform_thumbnail_map"]
            if sticker_composite_enabled():
                persist_keys.append("sticker_pack_json")
            if pikzels_render_failures:
                persist_keys.append("pikzels_render_failures")
            schedule_persist_artifact_now(ctx, *persist_keys)
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
        # Styled pixels did not ship — keep the sharpness-selected frame as the
        # canonical thumbnail path and R2 artifact (set before the styled block).
        ctx.thumbnail_path = best_path
        ctx.output_artifacts["thumbnail"] = str(best_path)
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

    # Optional fail-hard: when Pikzels is configured, reject template/raw-only thumbs
    # unless the user explicitly chose template/none pipeline (not stale auto prefs).
    try:
        require_studio = (
            os.environ.get("PIKZELS_REQUIRE_STUDIO_WHEN_KEY_CONFIGURED", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
    except Exception:
        require_studio = False
    if require_studio and studio_renderer_enabled():
        _rm = str(ctx.output_artifacts.get("thumbnail_render_method") or "").strip().lower()
        _pipe = normalize_thumbnail_render_pipeline(us)
        if _rm in ("template", "none", "") and _pipe not in ("template", "none"):
            raise ThumbnailError(
                "PIKZELS_API_KEY is configured but this upload did not produce a Pikzels studio "
                f"thumbnail (render_method={_rm or 'none'}). "
                f"Check studio_render_report skip_reason={studio_render_report.get('skip_reason')!r} "
                "or set thumbnailRenderPipeline=template to allow PIL-only."
            )

    return ctx
