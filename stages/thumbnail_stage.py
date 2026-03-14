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
from typing import Dict, List, Optional, Tuple

import httpx

from .context import JobContext, THUMBNAIL_BRIEF_PROMPT
from .errors import SkipStage

logger = logging.getLogger("uploadm8-worker.thumbnail")

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_THUMBNAIL_OFFSET = 1.0
MAX_THUMBNAIL_OFFSET     = 300.0
MIN_THUMB_SIZE           = 2048          # bytes — smaller = rejected
OPENAI_API_KEY           = os.environ.get("OPENAI_API_KEY", "")
OPENAI_THUMB_MODEL       = os.environ.get("OPENAI_THUMB_MODEL", "gpt-4o-mini")


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
    3-layer content category detection.
    Layer 1: user-provided caption/title hints
    Layer 2: filename keyword scan
    Layer 3: fall back to 'general' (GPT identifies from frames in the prompt)
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
        # Ensure platform_plan exists with defaults
        brief.setdefault("platform_plan", {})
        for plat in ("youtube", "instagram", "facebook", "tiktok"):
            brief["platform_plan"].setdefault(plat, {"enabled": True, "canvas": "16:9"})
        return brief
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

    headline = (brief.get("selected_headline") or brief.get("headline_options", [""])[0] or "").upper()[:30]
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

def _distribute_offsets(
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

    start  = clamp(duration * 0.05)
    end    = clamp(duration * 0.90)
    middle_count = n - 2
    step   = (end - start) / (middle_count + 1)
    middle = [clamp(start + step * (i + 1)) for i in range(middle_count)]

    return [start] + middle + [end]


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

    # ── Tier gates ──────────────────────────────────────────────────────────
    max_thumbnails = 1
    if ctx.entitlements:
        max_thumbnails = max(1, int(getattr(ctx.entitlements, "max_thumbnails", 1) or 1))

    ai_key_present = bool(OPENAI_API_KEY)
    can_ai = ai_key_present and bool(getattr(ctx.entitlements, "can_ai", False) if ctx.entitlements else False)

    # When AI is active, extract at least 4 frames so it has meaningful choices.
    # On free tier (can_ai=False), extraction_count == max_thumbnails.
    extraction_count = max(max_thumbnails, 4 if can_ai else 1)

    # User-specified manual offset (single-thumbnail mode only)
    raw_offset = (ctx.user_settings or {}).get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        user_offset = float(raw_offset)
        user_offset = max(0.0, min(user_offset, MAX_THUMBNAIL_OFFSET))
    except (TypeError, ValueError):
        user_offset = DEFAULT_THUMBNAIL_OFFSET

    # ── Category detection ──────────────────────────────────────────────────
    category = _detect_category(ctx)

    logger.info(
        f"Thumbnail stage: video={video_path.name}, "
        f"max_thumbnails={max_thumbnails}, extraction_count={extraction_count}, "
        f"category={category}, "
        f"ai={'enabled' if can_ai else ('no-key' if not ai_key_present else 'plan-gate')}"
    )

    # ── Duration probe ──────────────────────────────────────────────────────
    duration = await _get_video_duration(video_path)
    logger.debug(f"Video duration: {duration:.1f}s")

    # ── Distribute offsets ──────────────────────────────────────────────────
    offsets = _distribute_offsets(
        duration=duration,
        n=extraction_count,
        user_offset=user_offset if extraction_count == 1 else None,
    )
    logger.debug(f"Thumbnail offsets: {[f'{o:.1f}s' for o in offsets]}")

    # ── Extract and score all frames ────────────────────────────────────────
    candidates: List[Tuple[Path, float]] = []

    for idx, offset in enumerate(offsets):
        out_path = ctx.temp_dir / f"thumb_{ctx.upload_id}_{idx:02d}.jpg"
        success = await _extract_frame(video_path, out_path, offset)

        if not success and offset > 0:
            logger.debug(f"Frame at {offset:.1f}s failed — retrying at t=0")
            success = await _extract_frame(video_path, out_path, 0.0)

        if success:
            score = await _score_sharpness(out_path)
            if score == 0.0:
                score = out_path.stat().st_size / 1_000_000  # file-size proxy
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

    if can_ai and len(candidates) > 1:
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
        reason = "no API key" if not ai_key_present else (
            "plan gate" if not can_ai else "only 1 candidate"
        )
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

    # ── Styled thumbnails (MrBeast-style composite) — Trill + non-Trill, every upload ──
    # Gated by: can_custom_thumbnails + user pref styled_thumbnails (default True)
    can_custom = bool(getattr(ctx.entitlements, "can_custom_thumbnails", False) if ctx.entitlements else False)
    can_ai_style = bool(getattr(ctx.entitlements, "can_ai_thumbnail_styling", False) if ctx.entitlements else False)
    us = ctx.user_settings or {}
    styled_enabled = us.get("styled_thumbnails", us.get("styledThumbnails", True))
    if can_custom and styled_enabled and ctx.temp_dir:
        brief: Optional[Dict] = None
        if can_ai:
            brief = await _generate_thumbnail_brief(ctx, category)
        if not brief and can_ai:
            # Fallback brief when GPT fails — minimal defaults
            brief = {
                "selected_headline": (ctx.get_effective_title() or "WATCH")[:20].upper(),
                "headline_options": [],
                "badge_text": "NEW",
                "badge_style": "red",
                "directional_element": "circle",
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
                "directional_element": "circle",
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
        ctx.output_artifacts["thumbnail_brief_json"] = json.dumps(brief)

        # TikTok: no custom thumbnail via API — store thumb_offset for worker
        platform_map: Dict[str, str] = {}
        tiktok_plan = brief.get("platform_plan", {}).get("tiktok", {})
        ctx.output_artifacts["tiktok_thumb_offset_seconds"] = str(
            tiktok_plan.get("thumb_offset_seconds", 1.5)
        )

        # Render per platform (YouTube, Instagram, Facebook)
        platforms_to_render = [p for p in ("youtube", "instagram", "facebook")
                              if (brief.get("platform_plan", {}).get(p, {}).get("enabled", True))
                              and p in [pl.lower() for pl in (ctx.platforms or [])]]
        render_method = "none"
        primary_styled: Optional[Path] = None  # Prefer YouTube for primary
        for platform in platforms_to_render:
            out_name = f"thumb_styled_{platform}_{ctx.upload_id}.jpg"
            out_path = ctx.temp_dir / out_name
            ok = False
            if can_ai_style and OPENAI_API_KEY:
                ok = await _ai_edit_thumbnail(best_path, brief, out_path, retry_reduce=False)
                if not ok:
                    ok = await _ai_edit_thumbnail(best_path, brief, out_path, retry_reduce=True)
                if ok:
                    render_method = "ai_edit"
            if not ok:
                ok = _render_template_thumbnail(best_path, brief, platform, out_path)
                if ok:
                    render_method = "template"
            if ok:
                platform_map[platform] = str(out_path)
                if primary_styled is None or platform == "youtube":
                    primary_styled = out_path
        if primary_styled and primary_styled.exists():
            ctx.thumbnail_path = primary_styled
            ctx.output_artifacts["thumbnail"] = str(primary_styled)

        ctx.output_artifacts["thumbnail_render_method"] = render_method
        ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(platform_map)

    logger.info(
        f"Thumbnail stage complete: {len(candidates)} frames, "
        f"selected={best_path.name} "
        f"(method={selection_method}, sharpness={best_score:.4f})"
    )

    return ctx
