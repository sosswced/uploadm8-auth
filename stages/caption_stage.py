"""
UploadM8 Caption Stage — Universal Edition
===========================================
Generates AI-powered titles, captions, and hashtags using OpenAI GPT-4o.

Key features:
  - Universal content recognition engine — 3-layer detection:
      Layer 1: User hint (ctx.caption / ctx.title keyword scan)
      Layer 2: Filename signal scan (100+ keyword patterns)
      Layer 3: GPT-4o-mini vision confirmation (runs on actual frames via prompt)
    Covers: beauty, food, home renovation, gardening, fitness, fashion,
    gaming, travel, pets, education, comedy, tech, music, real estate,
    automotive, lifestyle/vlog, sports, ASMR, + unlimited general fallback.
    Each category carries tone guide, hook templates, and hashtag seeds.
  - Trill telemetry integration: when .map telemetry is present, speed/location/
    score bucket are injected as MUST-USE story beats.
  - Platform-aware content: TikTok/IG/FB get caption only; YouTube gets title+desc.
  - Frame count pulled from user settings (captionFrameCount, default 6, clamped 2-12).
  - _finalise_hashtags() is NOW CALLED after AI generation — blocked hashtags
    are NEVER written to ctx.ai_hashtags.
  - always_hashtags from user settings are enforced at two layers:
    caption stage (via _finalise_hashtags) and publish stage (via
    ctx.get_effective_hashtags), so they can never be silently dropped.

FIXES APPLIED:
  [ISSUE 3] _finalise_hashtags() is now invoked inside run_caption_stage()
            after every AI hashtag generation — blocked tags filtered,
            always_hashtags merged in, result capped at user max.
  [ISSUE 4] Universal content recognition engine added. Any content type
            (makeup, food, gaming, real estate, ASMR, gardening, etc.) now
            gets category-specific prompt context, tone guide, hook templates,
            and hashtag seeds that the AI uses to produce specific,
            algorithm-optimised content instead of generic fallback.

Exports: run_caption_stage(ctx)
"""

import os
import json
import base64
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

from core.helpers import coerce_hashtag_list, sanitize_hashtag_body, strip_stray_hashtag_json_blob

import httpx

from .errors import SkipStage, StageError, ErrorCode
from .context import (
    JobContext,
    build_fusion_summary_text,
    build_hydration_story_text,
    format_route_trill_hint,
    is_placeholder_upload_caption,
    is_placeholder_upload_title,
)
from .ai_service_costs import user_pref_ai_service_enabled
from services.pipeline_ai_trace import record_ai_pipeline_trace

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

# M8_ENGINE path: scene graph + user-seeded content_strategy → OpenAI JSON → rank → ctx.
# Set UPLOADM8_M8_CAPTION_ENGINE=false to force the legacy single-prompt narrative path only.
_USE_M8_CAPTION_ENGINE = os.environ.get("UPLOADM8_M8_CAPTION_ENGINE", "true").lower() in (
    "1", "true", "yes", "on",
)


def _trace_caption(ctx: Any, upload_id: str, event: str, payload: Dict[str, Any]) -> None:
    record_ai_pipeline_trace(ctx, upload_id, f"caption.{event}", payload, log=logger)

# Pricing per 1K tokens (gpt-4o-mini defaults; gpt-4o is ~10x)
COST_PER_1K_INPUT  = 0.000150
COST_PER_1K_OUTPUT = 0.000600
COST_PER_IMAGE     = 0.00765   # low-detail vision


def _setting_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
    return bool(value)


def _user_explicitly_disabled_auto_captions(us: Dict[str, Any]) -> bool:
    """
    True when upload preferences explicitly turn off auto captions.

    Used so placeholder title/caption hydration does not override a deliberate
    ``autoCaptions: false`` (or legacy string ``\"false\"``).
    """
    if "auto_generate_captions" in us and not _setting_bool(us.get("auto_generate_captions"), True):
        return True
    if "autoCaptions" in us and not _setting_bool(us.get("autoCaptions"), True):
        return True
    if "auto_captions" in us and not _setting_bool(us.get("auto_captions"), True):
        return True
    return False


# ============================================================
# Universal Content Category Engine
# ============================================================

# Category definitions:
#   keywords      : filename/title/caption signal words (lowercase)
#   tone          : default tone guidance injected into the AI prompt
#   hook_templates: example hooks to inspire the AI (it adapts, not copies)
#   hashtag_seeds : starting hashtag vocabulary (no # prefix — AI expands from here)
CONTENT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "automotive": {
        "keywords": [
            "car", "cars", "drive", "driving", "road", "highway", "speed",
            "mph", "truck", "suv", "motorcycle", "bike", "moto", "drift",
            "race", "track", "lap", "vehicle", "auto", "engine", "exhaust",
            "trill", "dashcam", "cruise", "roadtrip", "joyride", "throttle",
            "horsepower", "turbo", "supercar", "hypercar", "offroad", "jeep",
        ],
        "tone": (
            "Grounded driving content: prefer HUD speeds, place names, driver labels, "
            "and named tracks over generic petrolhead filler. Match intensity to the "
            "actual footage — a calm cruise stays calm; a real high-speed beat can surge."
        ),
        "hook_templates": [
            "This is what {speed} mph feels like 🔥",
            "The road doesn't care about your speed limit 💨",
            "Not all classrooms have four walls — some have four wheels",
        ],
        "hashtag_seeds": [
            "carporn", "carsofinstagram", "carlife", "autolife", "roadtrip",
            "dashcam", "drivingvlog", "carculture", "joyride", "speedlimit",
        ],
    },
    "beauty": {
        "keywords": [
            "makeup", "beauty", "skincare", "foundation", "concealer", "blush",
            "lipstick", "lipgloss", "eyeshadow", "mascara", "eyeliner", "brow",
            "contour", "highlight", "glam", "grwm", "get ready", "routine",
            "sephora", "ulta", "drugstore", "dupe", "glow", "bronzer",
            "primer", "serum", "moisturizer", "toner", "cleanser", "spf",
        ],
        "tone": (
            "Aspirational, empowering, and approachable. "
            "Tutorial vibes mixed with genuine confidence. Make it feel achievable."
        ),
        "hook_templates": [
            "You don't need expensive products to get this look ✨",
            "Here's what worked for this look 🌟",
            "GRWM for a night out 💄",
        ],
        "hashtag_seeds": [
            "makeuptutorial", "beautytips", "grwm", "skincareroutine", "glowup",
            "makeuplover", "beautyhacks", "skincarecheck", "beautyreview",
        ],
    },
    "food": {
        "keywords": [
            "food", "recipe", "cook", "cooking", "bake", "baking", "eat",
            "eating", "meal", "dinner", "lunch", "breakfast", "snack",
            "restaurant", "foodie", "chef", "kitchen", "taste", "delicious",
            "yummy", "tasty", "homemade", "ingredients", "dish", "dessert",
            "cake", "pasta", "steak", "sushi", "pizza", "burger", "salad",
        ],
        "tone": (
            "Warm, sensory, mouth-watering. "
            "Make viewers taste it through the screen. Recipes = instant saves."
        ),
        "hook_templates": [
            "I made this in under 20 minutes and it's incredible 🍽️",
            "The easiest recipe you'll make all week",
            "POV: you just discovered your new favourite meal",
        ],
        "hashtag_seeds": [
            "foodie", "foodtok", "recipe", "homecooking", "easyrecipes",
            "foodvideo", "cookingvideo", "mealprep", "whatieatinaday",
        ],
    },
    "home_renovation": {
        "keywords": [
            "reno", "renovation", "diy", "home", "house", "room", "makeover",
            "before after", "beforeafter", "transform", "decor", "interior",
            "design", "build", "construction", "fix", "repair", "tile",
            "paint", "floor", "wall", "cabinet", "remodel", "contractor",
            "woodwork", "carpentry", "plumbing", "electrical", "demo",
        ],
        "tone": (
            "Satisfying, process-driven. Before/after reveals. "
            "People love the transformation arc — tease the before, deliver the after."
        ),
        "hook_templates": [
            "This room cost $300 to transform — watch 👀",
            "Here's the before and after — transformation complete",
            "The before photos were giving me anxiety 😬",
        ],
        "hashtag_seeds": [
            "diyrenovation", "homeimprovement", "beforeandafter", "housereno",
            "interiordesign", "diyhome", "hometransformation", "hgtv",
        ],
    },
    "gardening": {
        "keywords": [
            "garden", "gardening", "plant", "plants", "grow", "growing",
            "flower", "flowers", "vegetable", "herb", "seed", "soil",
            "harvest", "compost", "greenhouse", "outdoor", "backyard",
            "nature", "green thumb", "bloom", "prune", "mulch", "raised bed",
        ],
        "tone": (
            "Calm, nurturing, educational. Community of plant people vibes. "
            "Patience and reward — the growth journey matters."
        ),
        "hook_templates": [
            "What 30 days of consistent watering looks like 🌱",
            "I almost killed this plant — here's what saved it",
            "Starting a garden from scratch: day 1",
        ],
        "hashtag_seeds": [
            "gardenlife", "plantlover", "growyourown", "gardentok",
            "urbangarden", "planttok", "homestead", "sustainableliving",
        ],
    },
    "fitness": {
        "keywords": [
            "workout", "gym", "fitness", "exercise", "train", "training",
            "lift", "lifting", "weights", "cardio", "run", "running",
            "yoga", "pilates", "hiit", "crossfit", "strength", "gains",
            "physique", "body", "health", "sweat", "reps", "sets", "pr",
            "personal record", "bulk", "cut", "shred", "lean",
        ],
        "tone": (
            "Motivational, raw, no-nonsense. "
            "Sweat is the currency here. Show the work, not just the results."
        ),
        "hook_templates": [
            "Nobody starts strong. They just start.",
            "What a consistent 90-day program actually looks like 💪",
            "The workout that changed how I train forever",
        ],
        "hashtag_seeds": [
            "fitness", "gymlife", "workout", "fitnessmotivation", "gymtok",
            "fitcheck", "workouttips", "strengthtraining", "cardio",
        ],
    },
    "fashion": {
        "keywords": [
            "fashion", "outfit", "ootd", "style", "clothes", "clothing",
            "wear", "wearing", "haul", "thrift", "thrifting", "fit check",
            "fitcheck", "lookbook", "trend", "streetwear", "streetstyle",
            "designer", "vintage", "aesthetic", "closet", "wardrobe",
        ],
        "tone": (
            "Confident, expressive, aesthetic. "
            "OOTD energy — let the fit do the talking. Style is identity."
        ),
        "hook_templates": [
            "OOTD that cost $12 from the thrift store 🔥",
            "Stop sleeping on this style combo",
            "The fit that broke the internet (in my mind at least)",
        ],
        "hashtag_seeds": [
            "ootd", "fashiontok", "outfitinspo", "stylecheck", "thrifted",
            "streetstyle", "lookbook", "fitcheck", "fashionadvice",
        ],
    },
    "gaming": {
        "keywords": [
            "game", "gaming", "gamer", "play", "playing", "gameplay",
            "stream", "twitch", "fps", "rpg", "mmorpg", "controller",
            "pc", "console", "xbox", "playstation", "nintendo", "fortnite",
            "minecraft", "valorant", "cod", "lol", "roblox", "speedrun",
            "clip", "montage", "ranked", "esports", "pro",
        ],
        "tone": (
            "Energetic, community-fluent. Speak gamer. "
            "Reactions, hype, and skill showcasing — the clip does the work."
        ),
        "hook_templates": [
            "The clip I've been waiting to post 🎮",
            "This strat worked in ranked 👀",
            "New personal best — I'm still shaking",
        ],
        "hashtag_seeds": [
            "gaming", "gamer", "gameplay", "gamingvideos", "gamertok",
            "pcgaming", "consolegaming", "gamingclips", "gamenight",
        ],
    },
    "travel": {
        "keywords": [
            "travel", "trip", "vacation", "holiday", "destination", "explore",
            "adventure", "abroad", "country", "city", "beach", "mountain",
            "hotel", "hostel", "airbnb", "flight", "backpack", "backpacking",
            "solo travel", "tourist", "sightseeing", "wanderlust", "passport",
        ],
        "tone": (
            "Wanderlust-inducing, vivid. "
            "Transport viewers there. FOMO is the goal — make them want to book a flight."
        ),
        "hook_templates": [
            "I moved to {location} for 30 days — here's what happened",
            "The destination nobody talks about 🌍",
            "Budget travel: {location} for under $50/day",
        ],
        "hashtag_seeds": [
            "travel", "travelgram", "traveltok", "wanderlust", "adventure",
            "exploremore", "solotravel", "travelvlog", "travelblogger",
        ],
    },
    "pets": {
        "keywords": [
            "dog", "cat", "pet", "puppy", "kitten", "animal", "furry",
            "paw", "tail", "bark", "meow", "bird", "fish", "hamster",
            "bunny", "rabbit", "vet", "adoption", "rescue", "cute",
            "doggo", "pupper", "floof", "doge",
        ],
        "tone": (
            "Wholesome, playful, emotional. "
            "Pets are content gold — let the animal be the star. "
            "One genuine moment beats a produced shoot every time."
        ),
        "hook_templates": [
            "My dog learned this in ONE day 🐶",
            "The moment this rescue dog realised he was home 🥹",
            "Nobody told me owning a {animal} would be like this",
        ],
        "hashtag_seeds": [
            "dogsofinstagram", "catsoftiktok", "petsoftiktok", "petlife",
            "dogmom", "catmom", "furryfriend", "adoptdontshop",
        ],
    },
    "education": {
        "keywords": [
            "learn", "learning", "teach", "tutorial", "how to", "howto",
            "tips", "tricks", "hacks", "explain", "explanation", "guide",
            "course", "study", "skill", "knowledge", "fact", "facts",
            "science", "history", "psychology", "money", "finance", "invest",
            "productivity", "life hack", "advice",
        ],
        "tone": (
            "Clear, credible, punchy. "
            "Teach one thing per video. The hook IS the insight — lead with the value."
        ),
        "hook_templates": [
            "Here's what actually works — based on what I tried",
            "I learned this the hard way so you don't have to",
            "One clear takeaway from this video",
        ],
        "hashtag_seeds": [
            "learnontiktok", "didyouknow", "educationalcontent", "lifelessons",
            "studytips", "knowledgebomb", "tipoftheday", "selfimprovement",
        ],
    },
    "comedy": {
        "keywords": [
            "funny", "comedy", "joke", "prank", "skit", "reaction", "meme",
            "lol", "laugh", "hilarious", "humor", "parody", "roast",
            "relatable", "pov", "nobody", "trend", "trending",
        ],
        "tone": (
            "Irreverent, fast, punchy. "
            "Don't explain the joke. Timing is everything. Less is always more."
        ),
        "hook_templates": [
            "POV: {relatable situation}",
            "Nobody:",
            "Tell me without telling me that you're a {thing}",
        ],
        "hashtag_seeds": [
            "funny", "comedy", "foryou", "relatable", "viral", "trending",
            "humor", "lol", "comedytok", "skit",
        ],
    },
    "tech": {
        "keywords": [
            "tech", "technology", "app", "software", "hardware", "phone",
            "iphone", "android", "laptop", "computer", "pc", "review",
            "unboxing", "setup", "desk setup", "battlestation", "ai",
            "programming", "code", "coding", "developer", "startup",
            "gadget", "gear", "product", "saas",
        ],
        "tone": (
            "Smart, practical, no fluff. "
            "Nerds and early adopters are the audience — respect their intelligence."
        ),
        "hook_templates": [
            "This AI tool saved me 3 hours today",
            "The gadget everyone will own in 2 years",
            "My honest review after 6 months of use",
        ],
        "hashtag_seeds": [
            "tech", "techtok", "technology", "techreview", "gadgets",
            "aitools", "productivity", "desksetup", "techhacks",
        ],
    },
    "music": {
        "keywords": [
            "music", "song", "singing", "sing", "cover", "original", "produce",
            "producing", "beat", "studio", "record", "recording", "guitar",
            "piano", "drums", "bass", "vocal", "lyrics", "concert", "gig",
            "mixtape", "album", "ep", "release",
        ],
        "tone": (
            "Raw, authentic, emotional. "
            "Let the music carry it — words are the backstage pass."
        ),
        "hook_templates": [
            "I wrote this song in 20 minutes — here's what came out",
            "This is my first original track 🎵",
            "The note that changes everything",
        ],
        "hashtag_seeds": [
            "music", "newmusic", "originalmusic", "musictok", "singerwriter",
            "indieartist", "songwriting", "studio", "musicproduction",
        ],
    },
    "real_estate": {
        "keywords": [
            "real estate", "property", "house", "home", "apartment", "condo",
            "listing", "for sale", "rent", "landlord", "tenant", "mortgage",
            "investing", "investment", "flip", "flipping", "airbnb",
            "cashflow", "rental", "equity", "roi",
        ],
        "tone": (
            "Authoritative, aspirational. Show the lifestyle. "
            "Numbers sell — lead with the ROI, close with the dream."
        ),
        "hook_templates": [
            "Here's what this property offers",
            "I bought my first rental property at 24 — here's how",
            "A closer look at this neighbourhood",
        ],
        "hashtag_seeds": [
            "realestate", "realestatelife", "househunting", "propertyinvesting",
            "realestateinvesting", "realtorlife", "homeseller", "houseflipping",
        ],
    },
    "sports": {
        "keywords": [
            "sport", "sports", "athlete", "athletic", "soccer", "football",
            "basketball", "baseball", "tennis", "golf", "swim", "swimming",
            "hockey", "cricket", "rugby", "mma", "boxing", "fight", "match",
            "game", "tournament", "league", "pro", "college", "training",
            "score", "goal", "point", "win", "championship",
        ],
        "tone": (
            "Competitive, electrifying. The scoreboard matters. "
            "Hype the achievement — make them feel the moment."
        ),
        "hook_templates": [
            "This goal took 3 years of practice 🥅",
            "Here's the play that got us to the finals",
            "The moment that defined this match",
        ],
        "hashtag_seeds": [
            "sports", "athlete", "sportsmotivation", "sportstok",
            "training", "competition", "sportsclips", "winning",
        ],
    },
    "asmr": {
        "keywords": [
            "asmr", "satisfying", "relaxing", "calm", "peaceful", "soothing",
            "triggers", "tingles", "whisper", "tapping", "crunchy", "slime",
            "soap", "cutting", "no talking", "oddly satisfying",
        ],
        "tone": (
            "Quiet, intimate, sensory. "
            "Don't oversell it — let the content breathe. Less = more in ASMR."
        ),
        "hook_templates": [
            "60 seconds of satisfying sounds 🎧",
            "No talking — just the triggers",
            "Instant calm — no talking",
        ],
        "hashtag_seeds": [
            "asmr", "asmrsounds", "satisfying", "relaxing", "asmrcommunity",
            "asmrvideo", "oddlysatisfying", "calming",
        ],
    },
    "lifestyle": {
        "keywords": [
            "vlog", "day in my life", "daily", "morning routine", "night routine",
            "productive", "productivity", "life update", "story time",
            "minimalist", "aesthetic", "wellness", "mental health",
            "self care", "journal", "gratitude", "haul", "unboxing",
        ],
        "tone": (
            "Personal, warm, authentic. "
            "Viewers want to hang out with you — let them in. Real > polished."
        ),
        "hook_templates": [
            "A realistic day in my life (not the highlight reel)",
            "I tried {challenge} for 30 days — here's what I noticed",
            "My morning routine — what works for me",
        ],
        "hashtag_seeds": [
            "dayinmylife", "vlog", "lifestylevlog", "morningroutine",
            "productivitytips", "selfcare", "wellnesstok", "authenticlife",
        ],
    },
    "general": {
        "keywords": [],  # catch-all — always matches last
        "tone": (
            "Engaging, authentic, and specific to what is actually visible in the video. "
            "Identify the content type from the frames and match your tone to it. "
            "Accuracy over hype — never overpromise or mislead."
        ),
        "hook_templates": [
            "Here's what actually happened",
            "What you see in this frame",
        ],
        "hashtag_seeds": [
            "viral", "fyp", "foryoupage", "trending", "mustwatch",
        ],
    },
}

# Detection priority order (more specific first, general last)
_CATEGORY_PRIORITY = [
    "automotive", "beauty", "food", "home_renovation", "gardening",
    "fitness", "fashion", "gaming", "travel", "pets", "education",
    "comedy", "tech", "music", "real_estate", "sports", "asmr", "lifestyle",
    "general",
]


# Legacy narrative-prompt directives — derived from core.caption_creative
# (single source). Do not hardcode parallel style/tone/voice lists here.
from core.caption_creative import (
    STYLE_DIRECTIVES as _CC_STYLE,
    TONE_DIRECTIVES as _CC_TONE,
    VOICE_DIRECTIVES as _CC_VOICE,
)

TONE_DIRECTIVES: Dict[str, str] = {
    k: str(v.get("register") or "") for k, v in _CC_TONE.items()
}
VOICE_PROFILES: Dict[str, str] = {
    k: str(v.get("persona") or "") for k, v in _CC_VOICE.items()
}
STYLE_DIRECTIVES: Dict[str, str] = {
    k: str(v.get("blueprint") or "") for k, v in _CC_STYLE.items()
}


def _detect_category_from_text(text: str) -> Optional[str]:
    """Scan a text string for category keyword signals. Returns first match or None."""
    if not text:
        return None
    text_lower = text.lower()
    for cat in _CATEGORY_PRIORITY:
        if cat == "general":
            continue
        for kw in CONTENT_CATEGORIES[cat].get("keywords", []):
            if kw in text_lower:
                return cat
    return None


def _detect_content_category(ctx: JobContext) -> str:
    """
    3-layer content category detection.

    Layer 1: User hint — ctx.caption and ctx.title keyword scan.
    Layer 2: Filename signal scan.
    Layer 3: Falls back to 'general'; the AI prompt instructs GPT to
             confirm from visual frames and adjust its output accordingly.

    Returns a key from CONTENT_CATEGORIES.
    """
    # Layer 1: user-provided hints
    for text in (ctx.caption, ctx.title):
        cat = _detect_category_from_text(text or "")
        if cat:
            logger.debug(f"Content category from user hint: {cat}")
            return cat

    # Layer 2: filename
    cat = _detect_category_from_text(ctx.filename or "")
    if cat:
        logger.debug(f"Content category from filename: {cat}")
        return cat

    logger.debug("Content category: general (GPT vision will identify from frames)")
    return "general"


def _build_category_context_block(category: str, location: Optional[str] = None) -> str:
    """
    Build the category-specific context block injected into the AI prompt.
    This gives GPT specific tone/hook/hashtag vocabulary instead of generic fallback.
    """
    cat_data = CONTENT_CATEGORIES.get(category, CONTENT_CATEGORIES["general"])
    tone = cat_data["tone"]
    hooks = cat_data.get("hook_templates", [])
    seeds = cat_data.get("hashtag_seeds", [])

    lines = [
        f"CONTENT CATEGORY DETECTED: {category.upper().replace('_', ' ')}",
        f"TONE GUIDE: {tone}",
    ]
    if hooks:
        hook_examples = " | ".join(f'"{h}"' for h in hooks[:2])
        lines.append(
            f"HOOK INSPIRATION (adapt freely, never copy verbatim; must accurately reflect visible content): {hook_examples}"
        )
    if seeds:
        lines.append(
            f"HASHTAG VOCABULARY (use as seeds, expand with more specific terms): "
            f"{', '.join(seeds[:10])}"
        )
    if location and category in ("automotive", "travel", "sports", "lifestyle", "real_estate"):
        lines.append(
            f"LOCATION: Content filmed near {location} — "
            f"weave naturally into caption and include as a hashtag."
        )

    block = "\n".join(f"  {ln}" for ln in lines)
    return (
        "━━ CONTENT CATEGORY CONTEXT ━━\n"
        f"{block}\n"
        "━━ Use this category context to produce SPECIFIC, niche-optimised content. "
        "Generic captions get buried by the algorithm. Specific, category-fluent "
        "captions get algorithmic lift. Hook must feel native to this content type "
        "and must accurately reflect what is visible — never overpromise or mislead. ━━"
    )


# ============================================================
# Frame Collection
# ============================================================

async def _collect_story_frames(ctx: JobContext, max_frames: int) -> List[Path]:
    """
    Collect frames for AI analysis.

    Priority:
      1. ctx.thumbnail_paths  (already extracted by thumbnail_stage, multi-frame)
      2. ctx.thumbnail_path   (single frame fallback)
      3. Live extraction from video (last resort)
    """
    if ctx.thumbnail_paths:
        existing = [p for p in ctx.thumbnail_paths if Path(p).exists()]
        if existing:
            existing.sort(key=lambda p: str(p))
            logger.debug(f"Using {len(existing)} pre-extracted frames from thumbnail_stage")
            return existing[:max_frames]

    if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
        logger.debug("Using single thumbnail_path for caption AI")
        return [ctx.thumbnail_path]

    video_path = ctx.local_video_path or ctx.processed_video_path
    if not video_path or not Path(video_path).exists():
        return []

    logger.debug("No pre-extracted frames — running live extraction for caption AI")
    return await _live_extract_frames(Path(video_path), ctx.temp_dir, max_frames)


async def _live_extract_frames(video_path: Path, temp_dir: Path, n: int) -> List[Path]:
    """Fallback: extract N frames from video on-the-fly."""
    frames: List[Path] = []
    try:
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
    """Construct the Trill story beat injection string (driving telemetry)."""
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
            lines.append(
                "Flagged: Excessive speed detected — convey raw energy "
                "without glorifying recklessness"
            )

    if td:
        if td.max_speed_mph and td.max_speed_mph > 0:
            lines.append(f"Peak speed: {td.max_speed_mph:.0f} mph")
        if td.avg_speed_mph and td.avg_speed_mph > 0:
            lines.append(f"Average speed: {td.avg_speed_mph:.0f} mph")
        if td.total_distance_miles and td.total_distance_miles > 0:
            lines.append(f"Distance covered: {td.total_distance_miles:.1f} miles")
        if td.euphoria_seconds and td.euphoria_seconds > 0:
            lines.append(f"Euphoria time: {td.euphoria_seconds:.0f}s sustained high speed")
        if td.location_display:
            lines.append(f"Location: {td.location_display}")
        if td.location_road:
            lines.append(f"Road/highway: {td.location_road}")
        if getattr(td, "location_country", None):
            lines.append(f"Country: {td.location_country}")
        mla, mlo = getattr(td, "mid_lat", None), getattr(td, "mid_lon", None)
        if mla is not None and mlo is not None:
            try:
                lines.append(f"GPS mid-route (WGS84): {float(mla):.5f}, {float(mlo):.5f}")
            except (TypeError, ValueError):
                pass
        sla, slo = getattr(td, "start_lat", None), getattr(td, "start_lon", None)
        if sla is not None and slo is not None:
            try:
                lines.append(f"GPS route start (WGS84): {float(sla):.5f}, {float(slo):.5f}")
            except (TypeError, ValueError):
                pass
        rh = format_route_trill_hint(getattr(td, "points", None) or [])
        if rh:
            lines.append(rh)

    if not lines:
        return None

    beat = "\n".join(f"  • {ln}" for ln in lines)
    return (
        "━━ TRILL STORY BEAT (MUST appear in caption) ━━\n"
        f"{beat}\n"
        "━━ Reference the score bucket personality, peak speed, and location. "
        "Make the caption feel lived-in and real with these actual numbers. ━━"
    )


# ============================================================
# Prompt Builder (category-aware)
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
    category: str,
) -> str:
    """Build the full AI prompt with category context, multi-frame narrative, and Trill beat."""
    platform_str = ", ".join(ctx.platforms) if ctx.platforms else "social media"
    caption_length = {
        "story":   "150–280 characters — tell a narrative arc",
        "punchy":  "under 120 characters — hook in the first 3 words",
        "factual": "100–200 characters — lead with the most impressive stat",
    }.get(caption_style, "150–280 characters")

    _allowed_plats = frozenset({"youtube", "tiktok", "instagram", "facebook"})
    _raw_plats = [str(p).strip().lower() for p in (ctx.platforms or []) if str(p).strip()]
    _selected_platforms = [p for p in _raw_plats if p in _allowed_plats]
    if not _selected_platforms:
        _selected_platforms = ["youtube", "tiktok", "instagram", "facebook"]
    sel_label = ", ".join(_selected_platforms)

    _platform_title_specs = {
        "youtube": "Keyword-front-loaded searchable title; strongest nouns in the first 40 chars; max 100 chars; plain language; no emojis; avoid Title Case Every Word and generic hooks (POV:, This is why, Wait for, Nobody expected).",
        "tiktok": "3–8 word FYP-style hook; concrete visual, place, or speed when visible; max 100 chars; no emoji; never invent facts.",
        "instagram": "Short Reels headline: aesthetic + specific to visible details; max 100 chars; no emoji.",
        "facebook": "Plain-language discovery title for a broad feed; who/what/where when visible; max 100 chars.",
    }
    _platform_caption_specs = {
        "youtube": f"YouTube — {caption_length}; searchable nouns early; evidence-grounded description energy.",
        "tiktok": f"TikTok — {caption_length}; hook in the first 3 words; trend-native but never invent facts.",
        "instagram": f"Instagram Reels — {caption_length}; niche keywords + vibe tied to visible details.",
        "facebook": f"Facebook — {caption_length}; conversational, shareable prose; clear who/what/where.",
    }

    # ── Task list (per-platform JSON + legacy top-level keys for back-compat) ─
    tasks: List[str] = []
    ti = 0
    if generate_title:
        ti += 1
        spec_lines = "\n".join(
            f"   • {pl.upper()}: {_platform_title_specs[pl]}"
            for pl in _selected_platforms
            if pl in _platform_title_specs
        )
        tasks.append(
            f"{ti}. titles_by_platform: JSON object with one string title per target platform.\n"
            f"   Target platforms for this upload: {sel_label}\n"
            f"   Each platform title MUST be meaningfully different from the others.\n"
            f"{spec_lines}\n"
            '   Also set top-level "title" to the same string as titles_by_platform["youtube"] '
            'when "youtube" is among the targets; otherwise use the best title for the first '
            "listed target platform."
        )
    if generate_caption:
        ti += 1
        cspec_lines = "\n".join(
            f"   • {pl.upper()}: {_platform_caption_specs[pl]}"
            for pl in _selected_platforms
            if pl in _platform_caption_specs
        )
        tasks.append(
            f"{ti}. captions_by_platform: JSON object with one caption string per target platform.\n"
            f"   Target platforms: {sel_label}\n"
            f"{cspec_lines}\n"
            '   Also set top-level "caption" to the same string as captions_by_platform["tiktok"] '
            'when "tiktok" is among the targets; otherwise use the caption for the first '
            "listed short-form platform (instagram, then facebook, else youtube)."
        )
    if generate_hashtags and hashtag_count > 0:
        ti += 1
        style_hint = {
            "trending": "prioritise viral trending tags",
            "niche":    "prioritise specific niche community tags",
            "mixed":    "mix viral and niche tags",
        }.get(hashtag_style, "mix viral and niche tags")
        tasks.append(
            f"{ti}. hashtags_by_platform: JSON object mapping each target platform to an array of "
            f"up to {hashtag_count} hashtag words WITHOUT the # symbol ({style_hint}). "
            f"NEVER return single letters or word fragments.\n"
            f"   Target platforms: {sel_label}\n"
            '   Also set top-level "hashtags" to the same array as hashtags_by_platform["tiktok"] '
            'when "tiktok" is among the targets; else hashtags_by_platform["instagram"], then '
            '"facebook", else "youtube".'
        )
    tasks_block = "\n".join(tasks) if tasks else "(none requested)"

    # ── Tone + voice (user prefs): composable layers on top of category context
    tone_instruction = TONE_DIRECTIVES.get(caption_tone) or TONE_DIRECTIVES["authentic"]

    us = getattr(ctx, "user_settings", None) or {}
    voice_key = str(us.get("captionVoice") or us.get("caption_voice") or "default").lower()
    if voice_key not in VOICE_PROFILES:
        voice_key = "default"
    voice_instruction = VOICE_PROFILES[voice_key]

    style_instruction = STYLE_DIRECTIVES.get(caption_style) or STYLE_DIRECTIVES["story"]

    # ── Multi-frame narrative instruction ────────────────────────────────────
    if num_frames > 1:
        frame_instruction = (
            f"You are being shown {num_frames} frames from this video in CHRONOLOGICAL ORDER "
            f"(frame 1 = start, frame {num_frames} = near end). "
            f"Use the visual progression to tell a STORY — how did it evolve? "
            f"What was the peak moment? "
            f"Identify everything visible: environment, subject, activity, products, "
            f"energy level, branding, skill level. Use ALL of it. "
            f"Generic = buried. Specific = algorithmic lift."
        )
    elif num_frames == 1:
        frame_instruction = (
            "You are being shown 1 frame. "
            "Identify everything visible: environment, subject, activity, products, energy, "
            "branding, skill level. Use all visible signals for specific content."
        )
    else:
        frame_instruction = (
            "No video frames are attached. Rely on the spoken transcript (if any), filename, "
            "user hints, and telemetry/context blocks only. Do not invent on-screen visuals."
        )

    # ── Context block ────────────────────────────────────────────────────────
    # Strip stray JSON-hashtag junk before showing the AI, otherwise the model
    # echoes it back into the freshly generated caption (the "nasty hashtags" bug).
    context_lines = [f"Filename: {ctx.filename}", f"Target platforms: {platform_str}"]
    hp_ctx = getattr(ctx, "hydration_payload", None)
    if isinstance(hp_ctx, dict) and str(hp_ctx.get("hydration_story") or "").strip():
        hydration_story = str(hp_ctx["hydration_story"]).strip()
    else:
        hydration_story = build_hydration_story_text(ctx, max_chars=700)
    if hydration_story:
        if not isinstance(getattr(ctx, "output_artifacts", None), dict):
            ctx.output_artifacts = {}
        ctx.output_artifacts["hydration_story"] = hydration_story
        context_lines.append(hydration_story)
    try:
        from stages.context import build_video_story_timeline

        _tl = build_video_story_timeline(ctx, max_events=12) or []
        _tl_bits: List[str] = []
        for ev in _tl[:8]:
            if not isinstance(ev, dict):
                continue
            txt = str(ev.get("text") or "").strip()
            if not txt:
                continue
            try:
                t_s = float(ev.get("t_seconds") if ev.get("t_seconds") is not None else 0)
            except (TypeError, ValueError):
                t_s = 0.0
            _tl_bits.append(f"t={t_s:.0f}s {txt[:120]}")
        if _tl_bits:
            context_lines.append("Timeline spine: " + " | ".join(_tl_bits))
    except Exception:
        pass
    title_hint = strip_stray_hashtag_json_blob(str(ctx.title or ""))
    if title_hint:
        context_lines.append(f"User title hint: {title_hint}")
    caption_hint = strip_stray_hashtag_json_blob(str(ctx.caption or ""))
    if caption_hint:
        context_lines.append(f"User caption hint: {caption_hint}")
    if ctx.location_name:
        context_lines.append(f"Filming location: {ctx.location_name}")
    context_block = "\n".join(f"• {ln}" for ln in context_lines)

    # ── Memory examples (few-shot from past uploads) ─────────────────────────
    memory_block = ""
    examples = getattr(ctx, "caption_memory_examples", None) or []
    if examples:
        lines = []
        for i, ex in enumerate(examples[:5], 1):
            t = strip_stray_hashtag_json_blob(str(ex.get("ai_title") or ""))[:120]
            c = strip_stray_hashtag_json_blob(str(ex.get("ai_caption") or ""))[:320]
            tags = ex.get("ai_hashtags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            tag_list = coerce_hashtag_list(tags)
            if tag_list:
                tag_str = ", ".join(str(x).lstrip("#") for x in tag_list[:15])
            else:
                tag_str = ""
            lines.append(f"Example {i} — title: {t}")
            if c:
                lines.append(f"  caption: {c}")
            if tag_str:
                lines.append(f"  hashtags: {tag_str}")
        memory_block = (
            "\n━━ PAST EXAMPLES (match voice but write fresh — do not copy verbatim) ━━\n"
            + "\n".join(lines)
            + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    # ── Category context block ────────────────────────────────────────────────
    category_block = _build_category_context_block(category, ctx.location_name)

    raw_tx = getattr(ctx, "ai_transcript", None)
    transcript_block = ""
    if raw_tx and str(raw_tx).strip():
        transcript_block = (
            "\n━━ SPOKEN CONTENT (speech-to-text — factual; do not contradict) ━━\n"
            f"{str(raw_tx).strip()[:6000]}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    vision_block = ""
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict) and vc and not vc.get("skipped"):
        bits: List[str] = []
        try:
            _mf = int(vc.get("vision_multi_frame") or 1)
        except (TypeError, ValueError):
            _mf = 1
        if _mf > 1:
            bits.append(f"Vision samples merged along clip: {_mf} frames (OCR blocks may be separated by ---).")
        ocr = (vc.get("ocr_text") or "").strip()
        if ocr:
            bits.append(f"On-screen / OCR text: {ocr[:2200]}")
        labels = vc.get("label_names") or []
        if labels:
            from core.vision_labels import filter_vision_labels_for_context

            cat = str(
                getattr(ctx, "thumbnail_category", None)
                or (getattr(ctx, "hydration_payload", None) or {}).get("category")
                or "general"
            )
            filtered = filter_vision_labels_for_context(
                labels,
                category=cat,
                filename=str(getattr(ctx, "filename", "") or ""),
            )
            if filtered:
                bits.append("Scene labels: " + ", ".join(str(x) for x in filtered[:22]))
        fc = vc.get("face_count")
        if fc:
            bits.append(f"Approx. faces in sampled frame: {fc}")
        lm = vc.get("landmark_names") or []
        if isinstance(lm, list) and lm:
            bits.append("Landmarks (Google Vision): " + ", ".join(str(x) for x in lm[:8]))
        logos = vc.get("logo_names") or []
        if isinstance(logos, list) and logos:
            bits.append("Logos / brands (Google Vision): " + ", ".join(str(x) for x in logos[:8]))
        if bits:
            vision_block = (
                "\n━━ GOOGLE VISION HYDRATION (always-on sampled frames) ━━\n"
                + "\n".join(bits)
                + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    scene_understanding_block = ""
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict) and (vu.get("scene_description") or vu.get("title_suggestion")):
        sd = str(vu.get("scene_description") or "").strip()
        ts = str(vu.get("title_suggestion") or "").strip()
        scene_understanding_block = (
            "\n━━ SCENE UNDERSTANDING (full video) ━━\n"
            f"{sd[:4000]}\n"
            + (f"Suggested title angle: {ts}\n" if ts else "")
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    video_intel_block = ""
    vi_ctx = getattr(ctx, "video_intelligence_context", None) or {}
    if isinstance(vi_ctx, dict) and not vi_ctx.get("error"):
        machine = (
            vi_ctx.get("machine_summary") or vi_ctx.get("summary_text") or ""
        ).strip()
        # Prefer structured tracks over dump strings for the prompt.
        label_bits: List[str] = []
        for row in (vi_ctx.get("segment_labels") or [])[:12]:
            if isinstance(row, dict) and row.get("description"):
                label_bits.append(str(row["description"]))
        logo_bits = [
            str(l.get("description") or "")
            for l in (vi_ctx.get("logos") or [])[:6]
            if isinstance(l, dict) and l.get("description")
        ]
        ocr_bits = [
            str(t.get("text") or "")[:40]
            for t in (vi_ctx.get("on_screen_text") or [])[:4]
            if isinstance(t, dict) and t.get("text")
        ]
        parts: List[str] = []
        if logo_bits:
            parts.append("Logos: " + ", ".join(logo_bits))
        if ocr_bits:
            parts.append("On-screen text: " + ", ".join(ocr_bits))
        if label_bits:
            parts.append(
                "Detector labels (internal only — never paste into title/caption): "
                + ", ".join(label_bits[:10])
            )
        if not parts and machine and "Video Intelligence" not in machine:
            parts.append(machine[:800])
        if parts:
            video_intel_block = (
                "\n━━ GOOGLE VIDEO INTELLIGENCE (evidence only — do NOT quote "
                "'Video Intelligence', 'labels:', or 'objects:' in title/caption) ━━\n"
                + "\n".join(parts)[:2200]
                + "\nWrite specific, human titles/captions with depth — never generic "
                "words like car/vehicle/highway/windshield alone.\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    audio_enrich_block = ""
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        gas = (ac.get("gpt_audio_summary") or "").strip()
        if gas:
            audio_enrich_block += (
                f"\n━━ AUDIO CONTENT SUMMARY ━━\n{gas[:1200]}\n━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        he = ac.get("hume_emotions")
        if isinstance(he, dict) and he.get("dominant_emotion"):
            audio_enrich_block += (
                f"\n━━ VOCAL EMOTION SIGNALS ━━\nDominant: {he.get('dominant_emotion')} "
                f"(score {he.get('dominant_score', '')}); intensity: {he.get('emotional_intensity', '')}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        mus_lines: List[str] = []
        if ac.get("music_detected") or (ac.get("music_title") or ac.get("music_artist")):
            if ac.get("music_artist"):
                mus_lines.append(f"Artist: {str(ac.get('music_artist')).strip()[:200]}")
            if ac.get("music_title"):
                mus_lines.append(f"Title: {str(ac.get('music_title')).strip()[:200]}")
            if ac.get("music_genre"):
                mus_lines.append(f"Genre: {str(ac.get('music_genre')).strip()[:120]}")
            if ac.get("copyright_risk"):
                mus_lines.append("Rights: third-party / catalogue match — do not claim you wrote this track.")
        if mus_lines:
            audio_enrich_block += (
                "\n━━ MUSIC RECOGNITION (ACRCloud / catalogue — factual) ━━\n"
                + "\n".join(mus_lines)
                + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        fn = str(ac.get("fusion_narrative") or "").strip()
        if fn:
            audio_enrich_block += (
                f"\n━━ AUDIO FUSION (one-line thematic anchor) ━━\n{fn[:1400]}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        tsc = str(ac.get("top_sound_class") or "").strip()
        spf = str(ac.get("sound_profile") or "").strip()
        if tsc or spf:
            bits = []
            if tsc:
                bits.append(f"Top sound class: {tsc}")
            if spf:
                bits.append(spf[:700])
            audio_enrich_block += (
                "\n━━ ENVIRONMENTAL AUDIO (classifiers) ━━\n" + "\n".join(bits) + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        yev = ac.get("yamnet_events")
        if isinstance(yev, list) and yev:
            audio_enrich_block += (
                "\n━━ AUDIO EVENTS (YAMNet / AudioSet slugs) ━━\n"
                + ", ".join(str(x) for x in yev[:16])
                + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    # ── Trill beat ───────────────────────────────────────────────────────────
    trill_beat = _build_trill_beat(ctx)
    trill_section = f"\n\n{trill_beat}" if trill_beat else ""

    # ── Platform-specific output notes ───────────────────────────────────────
    platform_notes = []
    if "youtube" in (ctx.platforms or []):
        platform_notes.append("YouTube: titles_by_platform.youtube is the SEO title.")
    if any(p in (ctx.platforms or []) for p in ("tiktok", "instagram", "facebook")):
        platform_notes.append(
            "TikTok/Instagram/Facebook: publish may be caption-led; still supply distinct titles per platform."
        )
    platform_notes.append(
        f"Per-platform JSON keys required only for: {sel_label}. Omit other platform keys entirely."
    )
    platform_note = " | ".join(platform_notes)

    ex_titles_shape = "{" + ", ".join(f'"{p}":"…"' for p in _selected_platforms) + "}"
    ex_caps_shape = "{" + ", ".join(f'"{p}":"…"' for p in _selected_platforms) + "}"
    ex_tags_shape = "{" + ", ".join(f'"{p}":["tag"]' for p in _selected_platforms) + "}"

    prompt = f"""You are helping polish upload copy for {platform_str} — write like the creator would,
from what is actually in the video and context blocks (not like a corporate social team or a generic "expert" persona).

{frame_instruction}
{memory_block}
{category_block}{transcript_block}{vision_block}{scene_understanding_block}{video_intel_block}{audio_enrich_block}{trill_section}

TONE DIRECTIVE ({caption_tone.upper()}): {tone_instruction}
VOICE PROFILE ({voice_key.upper()}): {voice_instruction}
CAPTION STYLE ({caption_style.upper()} — {caption_length}): {style_instruction} Follow this structure strictly.
{f"PLATFORM NOTE: {platform_note}" if platform_note else ""}

HOW TONE + VOICE + CATEGORY FIT TOGETHER:
- Category block above = subject vocabulary, hook patterns, and hashtag seeds for the detected vertical (or general).
- Tone = energy, pacing, and emotional register.
- Voice = who is speaking (persona and sentence habits).
- All three must agree with what is actually visible or said. If category tone and user tone differ, keep user tone and voice for delivery but pull facts and niche words only from evidence—never force a vertical that the footage does not support.

Generate the following for this video:
{tasks_block}

Context:
{context_block}

Rules:
- Content must feel AUTHENTIC — not AI-generated; avoid slogan voice and "brand manager" tone
- NEVER invent events, locations, or narratives not supported by the frames, transcript, or telemetry provided. If no frames: do not claim specific visuals.
- ACCURACY OVER ENGAGEMENT: Do NOT use clickbait patterns ("Nobody expected", "You need to see this", "The secret nobody tells you"). Describe what is actually shown. Hooks must reflect visible content — never overpromise or mislead.
- Hook in the first 3 words for short-form platforms
- Do not use emojis, emoticons, or decorative Unicode symbols in the title or caption
- HASHTAGS: each must be a complete word (e.g. "makeuptutorial", "gardenlife", "dashcam")
  NEVER return single characters or word fragments
- NEVER put hashtags, JSON arrays, escaped quotes, or "#word" tokens inside "caption" —
  all tags go ONLY in the "hashtags" array as plain words (no # prefix)
- AUDIO + SPEECH: When SPOKEN CONTENT or AUDIO blocks above exist, the caption MUST reflect
  real dialogue, named topics, or emotional beats from the transcript (paraphrase; do not invent lines).
  When MUSIC RECOGNITION lists artist/title/genre, weave 1–2 factual references into the caption and
  include 1–3 niche hashtag tokens derived from that metadata (no false ownership if rights note appears).
  When ENVIRONMENTAL AUDIO / AUDIO EVENTS exist, add concrete ambient cues (crowd, engine, rain, studio, etc.)
  in prose and hashtags where they improve specificity — never generic filler unrelated to those signals.
- Be SPECIFIC to what is actually visible — generic content gets buried
- If Trill data provided: caption MUST reference at least one real data point

Respond ONLY in this exact JSON format (no markdown, no explanation).
Include only the keys you are generating; use null for omitted top-level keys.
Per-platform objects MUST only include keys from this upload's target list: {sel_label}.

Example shape (ellipsis not literal):
{{"titles_by_platform": {ex_titles_shape}, "captions_by_platform": {ex_caps_shape}, "hashtags_by_platform": {ex_tags_shape}, "title": "…", "caption": "…", "hashtags": ["word1", "word2"]}}

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
    """Call OpenAI with prompt + attached frames. Returns title/caption/hashtags + per-platform maps."""
    result: Dict[str, Any] = {
        "title": None,
        "caption": None,
        "hashtags": [],
        "titles_by_platform": {},
        "captions_by_platform": {},
        "hashtags_by_platform": {},
        "tokens": {},
    }

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping AI content generation")
        return result

    content: List[Dict] = [{"type": "text", "text": prompt}]

    for frame in frames[:6]:
        try:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
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
                logger.error(f"OpenAI API error: {response.status_code} — {response.text[:300]}")
                return result

            data = response.json()
            usage = data.get("usage", {})
            result["tokens"] = {
                "prompt":     usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            }

            answer = data["choices"][0]["message"]["content"]
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

            _plat_ok = frozenset({"youtube", "tiktok", "instagram", "facebook"})

            tbp = parsed.get("titles_by_platform")
            if isinstance(tbp, dict):
                for k, v in tbp.items():
                    kk = str(k).strip().lower()
                    if kk in _plat_ok and v is not None and str(v).strip():
                        result["titles_by_platform"][kk] = str(v).strip()[:120]

            cbp = parsed.get("captions_by_platform")
            if isinstance(cbp, dict):
                for k, v in cbp.items():
                    kk = str(k).strip().lower()
                    if kk in _plat_ok and v is not None and str(v).strip():
                        result["captions_by_platform"][kk] = strip_stray_hashtag_json_blob(
                            str(v).strip()
                        )[:2200]

            hbp = parsed.get("hashtags_by_platform")
            if isinstance(hbp, dict):
                for k, v in hbp.items():
                    kk = str(k).strip().lower()
                    if kk not in _plat_ok or v is None:
                        continue
                    raw_pl = v if isinstance(v, list) else []
                    if isinstance(v, str):
                        raw_pl = [t.strip() for t in v.replace(",", " ").split() if t.strip()]
                    cleaned_pl: List[str] = []
                    for tag in raw_pl:
                        ts = str(tag).strip() if tag is not None else ""
                        if not ts:
                            continue
                        parts = ts.split() if " " in ts else [ts]
                        for part in parts:
                            body = sanitize_hashtag_body(str(part).strip())
                            if len(body) >= 2:
                                cleaned_pl.append(body)
                    result["hashtags_by_platform"][kk] = cleaned_pl[:hashtag_count]

            raw_tags = parsed.get("hashtags")
            if raw_tags is not None:
                if isinstance(raw_tags, str):
                    raw_tags = [t.strip() for t in raw_tags.replace(",", " ").split() if t.strip()]
                elif not isinstance(raw_tags, list):
                    raw_tags = []

                cleaned: List[str] = []
                for tag in raw_tags:
                    tag = str(tag).strip()
                    if not tag:
                        continue
                    parts = tag.split() if " " in tag else [tag]
                    for part in parts:
                        body = sanitize_hashtag_body(part)
                        if len(body) >= 2:
                            cleaned.append(body)

                result["hashtags"] = cleaned[:hashtag_count]
                logger.info(f"AI hashtags raw ({len(result['hashtags'])}): {result['hashtags']}")

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")

    return result


# Platform-meta tags stripped before M8 final hashtag merge (see signal_hashtags._BLOCKED_META).
_META_HASHTAG_SLUGS = frozenset({
    "viral", "fyp", "foryou", "foryoupage", "trending", "mustwatch",
    "follow", "like", "subscribe", "video", "reels", "content",
    "youtube", "tiktok", "instagram", "facebook",
})


def strip_meta_hashtags(
    tags: List[str],
    max_count: int,
    *,
    category: str = "general",
    extra_seeds: Optional[List[str]] = None,
) -> List[str]:
    """Remove platform-meta spam; keep evidence/category tags up to max_count."""
    _ = category, extra_seeds  # reserved for future seed-priority ordering
    out: List[str] = []
    seen: set = set()
    for raw in tags or []:
        body = sanitize_hashtag_body(str(raw))
        if not body or body in seen:
            continue
        if body.lower() in _META_HASHTAG_SLUGS:
            continue
        seen.add(body)
        out.append(body)
        if len(out) >= max(0, int(max_count or 0)):
            break
    return out[: max(0, int(max_count or 0))]


# ============================================================
# Hashtag Finalisation — FIX ISSUE 3: Now actually called
# ============================================================

def _finalise_hashtags(
    ai_tags: List[str],
    base_tags: List[str],
    blocked: List[str],
    max_total: int,
) -> List[str]:
    """
    Merge AI tags onto base/always tags, remove blocked, cap at max_total.

    FIXED: This function was previously defined but NEVER called inside
    run_caption_stage(). Blocked hashtags the user explicitly banned were
    appearing in every post. Now called after every AI generation.

    Order: base_tags (always_hashtags + preset) FIRST, then AI additions.
    Blocked tags are filtered from every position in the merged list.
    """
    blocked_set: set = set()
    for t in blocked or []:
        b = sanitize_hashtag_body(str(t))
        if b:
            blocked_set.add(b)
    seen: set = set()
    merged: List[str] = []

    for tag in list(base_tags or []) + list(ai_tags or []):
        body = sanitize_hashtag_body(tag)
        if not body or body in seen or body in blocked_set:
            continue
        seen.add(body)
        merged.append(f"#{body}")

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


def _context_engine_caption_fallback(ctx: JobContext, category: str) -> str:
    """
    Non-AI fallback caption for disabled plans/preferences.
    Uses fused context (scene/vision/audio/telemetry) so we avoid generic boilerplate.
    """
    summary = (build_fusion_summary_text(ctx) or "").strip()
    if summary:
        clean = " ".join(summary.split())
        return clean[:500]

    tel = ctx.telemetry_data or ctx.telemetry
    bits: List[str] = []
    title = (ctx.get_effective_title() or "").strip()
    if title:
        bits.append(title[:120])
    if tel and getattr(tel, "location_display", None):
        bits.append(f"near {str(tel.location_display).strip()[:80]}")
    if tel and (getattr(tel, "max_speed_mph", 0.0) or 0) > 0:
        bits.append(f"peak {float(tel.max_speed_mph):.0f} mph")
    if category and category != "general":
        bits.append(category.replace("_", " "))
    out = " | ".join([b for b in bits if b]).strip(" |")
    return (out or "New upload").strip()[:500]


def _context_hydration_allowed(ctx: JobContext, us: Dict[str, Any]) -> bool:
    """
    Gate non-AI caption hydration behind tier/entitlements + user preference.
    """
    raw = us.get("captionContextHydrationEnabled")
    if raw is None:
        raw = us.get("caption_context_hydration_enabled")
    if isinstance(raw, str):
        raw = raw.strip().lower() not in ("false", "0", "no", "off", "")
    if raw is not None and not bool(raw):
        return False

    ent = getattr(ctx, "entitlements", None)
    if not ent:
        return False

    tier = str(getattr(ent, "tier", "free") or "free").lower().strip()
    is_internal = bool(getattr(ent, "is_internal", False))
    max_frames = int(getattr(ent, "max_caption_frames", 0) or 0)

    # Paid tiers (creator_lite+) and internal tiers can use context hydration.
    # This keeps free-tier fallback behavior simple while still allowing explicit
    # overrides via entitlement changes.
    paid_like = tier in ("creator_lite", "creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin", "launch")
    return bool(is_internal or paid_like or max_frames >= 5)


# ============================================================
# Stage Entry Point
# ============================================================

async def run_caption_stage(ctx: JobContext, db_pool=None) -> JobContext:
    """
    Execute AI caption generation stage.

    Reads from ctx:
      - ctx.entitlements           — plan gates (can_ai, max_caption_frames)
      - ctx.user_settings          — all user preferences, loaded from
                                     users.preferences JSONB by db.load_user_settings()
      - ctx.thumbnail_paths        — multi-frame story frames (from thumbnail_stage)
      - ctx.trill_score / ctx.telemetry_data — Trill telemetry if applicable

    Writes to ctx:
      - ctx.ai_title
      - ctx.ai_caption
      - ctx.ai_hashtags   — finalised: blocked tags removed, always_hashtags merged in
    """
    # ── Plan gate ────────────────────────────────────────────────────────────
    if not ctx.entitlements:
        raise SkipStage("No entitlements")

    can_ai = getattr(ctx.entitlements, "can_ai", False)

    us = ctx.user_settings or {}

    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    caption_llm_enabled = user_pref_ai_service_enabled(
        us, "caption_llm", default=True, allowed_services=tier_allowed_set
    )

    # ── User preference toggles ──────────────────────────────────────────────
    generate_caption  = _setting_bool(us.get("autoCaptions", us.get("auto_captions")), False)
    generate_title    = generate_caption
    generate_hashtags = _setting_bool(us.get("aiHashtagsEnabled", us.get("ai_hashtags_enabled")), False)

    if us.get("auto_generate_captions") is not None:
        generate_caption = _setting_bool(us["auto_generate_captions"], False)
        generate_title   = generate_caption
    if us.get("auto_generate_hashtags") is not None:
        generate_hashtags = _setting_bool(us["auto_generate_hashtags"], False)

    # Upload-page title/caption are authoritative unless the title is a client
    # placeholder; placeholders must not block M8 from writing a hydrated title.
    title_is_placeholder = is_placeholder_upload_title(ctx.title or "", ctx.filename or "")
    caption_is_placeholder = is_placeholder_upload_caption(ctx.caption or "")
    if (ctx.title or "").strip() and not title_is_placeholder:
        generate_title = False
    elif title_is_placeholder:
        generate_title = True
    if (ctx.caption or "").strip() and not caption_is_placeholder:
        generate_caption = False
    elif caption_is_placeholder and title_is_placeholder and not _user_explicitly_disabled_auto_captions(
        us
    ):
        # Both placeholders → hydrate caption for empty uploads when the user
        # has not explicitly opted out of auto captions (missing pref uses this path).
        generate_caption = True

    # ── Style/tone preferences ───────────────────────────────────────────────
    from core.caption_creative import (
        normalize_caption_style,
        normalize_caption_tone,
        normalize_caption_voice,
    )

    caption_style = normalize_caption_style(us.get("captionStyle") or us.get("caption_style"))
    caption_tone = normalize_caption_tone(us.get("captionTone") or us.get("caption_tone"))
    caption_voice = normalize_caption_voice(us.get("captionVoice") or us.get("caption_voice"))

    hashtag_style = str(us.get("aiHashtagStyle") or us.get("ai_hashtag_style") or "mixed").lower()
    if hashtag_style not in (
        "lowercase",
        "capitalized",
        "camelcase",
        "mixed",
        "trending",
        "niche",
    ):
        hashtag_style = "mixed"

    # "Number of AI Hashtags" — prefer ai_hashtag_count; fall back to max_hashtags for legacy rows.
    try:
        raw_n = us.get("aiHashtagCount")
        if raw_n is None:
            raw_n = us.get("ai_hashtag_count")
        if raw_n is None or (isinstance(raw_n, str) and not str(raw_n).strip()):
            raw_n = us.get("maxHashtags") or us.get("max_hashtags")
        pref_max = int(raw_n or 5)
    except (TypeError, ValueError):
        pref_max = 5
    pref_max = max(1, min(pref_max, 50))
    hashtag_count = pref_max if generate_hashtags else 0

    model = str(us.get("trillOpenaiModel") or us.get("openai_model") or OPENAI_MODEL_DEFAULT)

    # ── Frame count: entitlement ceiling × user setting, clamped 2–12 ────────
    max_caption_frames = getattr(ctx.entitlements, "max_caption_frames", 6) or 6
    user_frame_count = int(
        us.get("captionFrameCount") or us.get("caption_frame_count") or max_caption_frames
    )
    num_frames = max(2, min(min(user_frame_count, max_caption_frames), 12))

    # ── Detect content category (3-layer: hint → filename → general) ─────────
    hp0 = getattr(ctx, "hydration_payload", None)
    if isinstance(hp0, dict) and str(hp0.get("category") or "").strip():
        category = str(hp0["category"]).strip().lower()
        category_source = str(hp0.get("category_source") or "payload")
    else:
        category = _detect_content_category(ctx)
        category_source = "caption_detector"
    logger.info(
        "Caption stage: category=%r source=%s upload_id=%s",
        category,
        category_source,
        str(getattr(ctx, "upload_id", "") or ""),
    )
    ctx.thumbnail_category = category

    # Re-roll recognition narrative with detected niche (food, travel, automotive, …).
    try:
        from services.google_visual_recognition import attach_visual_recognition

        attach_visual_recognition(ctx)
    except Exception:
        pass

    # If AI path is unavailable/disabled, still hydrate a caption from context engine.
    # This avoids stale generic defaults from older fallback copy.
    ai_path_enabled = bool(
        can_ai and caption_llm_enabled and (generate_title or generate_caption or generate_hashtags)
    )
    _trace_caption(ctx, str(ctx.upload_id), "gate", {
        "can_ai": bool(can_ai),
        "caption_llm_enabled": bool(caption_llm_enabled),
        "generate_title": bool(generate_title),
        "generate_caption": bool(generate_caption),
        "generate_hashtags": bool(generate_hashtags),
        "model": model,
        "m8_enabled": bool(_USE_M8_CAPTION_ENGINE),
        "category": category,
        "category_source": category_source,
    })
    if not ai_path_enabled:
        if generate_caption and not (ctx.caption or "").strip() and _context_hydration_allowed(ctx, us):
            ctx.ai_caption = _context_engine_caption_fallback(ctx, category)
            logger.info(
                "Caption stage: AI disabled (plan/prefs). Using context-engine fallback caption: %s",
                (ctx.ai_caption or "")[:120],
            )
        return ctx

    # ── Retrieve few-shot examples from upload_caption_memory (optional) ───
    if db_pool:
        try:
            from . import db as _db_stage
            ctx.caption_memory_examples = await _db_stage.fetch_caption_memory_examples(
                db_pool, str(ctx.user_id), category, limit=3
            )
        except Exception as _mem_err:
            logger.warning("Caption memory fetch skipped: %s", _mem_err)

    logger.info(
        f"Caption stage: category={category}, style={caption_style}, tone={caption_tone}, "
        f"voice={caption_voice}, hashtag_style={hashtag_style}, count={hashtag_count}, "
        f"model={model}, frames={num_frames}, m8_engine={_USE_M8_CAPTION_ENGINE}"
    )

    # ── Read hashtag enforcement settings ────────────────────────────────────
    raw_always = us.get("alwaysHashtags") or us.get("always_hashtags") or []
    if isinstance(raw_always, str):
        raw_always = [t.strip() for t in raw_always.replace(",", " ").split() if t.strip()]
    always_tags: List[str] = list(raw_always) if isinstance(raw_always, list) else []

    raw_blocked = us.get("blockedHashtags") or us.get("blocked_hashtags") or []
    if isinstance(raw_blocked, str):
        raw_blocked = [t.strip() for t in raw_blocked.replace(",", " ").split() if t.strip()]
    blocked_tags: List[str] = list(raw_blocked) if isinstance(raw_blocked, list) else []

    try:
        from .content_strategy import build_content_strategy

        strategy = build_content_strategy(
            ctx,
            category=category,
            user_caption_style=caption_style,
            user_caption_tone=caption_tone,
            user_caption_voice=caption_voice,
        )

        # Optional SerpAPI / YouTube title trend sample for M8 (geo-aware query).
        try:
            from stages.trend_intel import fetch_trend_intel, trend_intel_runtime_available

            if trend_intel_runtime_available():
                await fetch_trend_intel(ctx, category=category)
        except Exception as te:
            logger.warning("trend_intel skipped: %s", te)

        # ── Collect story frames ─────────────────────────────────────────────
        frames = await _collect_story_frames(ctx, num_frames)
        if not frames:
            logger.warning("No frames available — generating captions without visual context")

        used_m8 = False
        if _USE_M8_CAPTION_ENGINE:
            try:
                from .m8_engine import run_m8_caption_engine

                meta = await run_m8_caption_engine(
                    ctx,
                    frames=frames,
                    category=category,
                    caption_style=caption_style,
                    caption_tone=caption_tone,
                    caption_voice=caption_voice,
                    hashtag_style=hashtag_style,
                    hashtag_count=hashtag_count,
                    generate_title=generate_title,
                    generate_caption=generate_caption,
                    generate_hashtags=generate_hashtags,
                    model=model,
                    blocked_tags=blocked_tags,
                    always_tags=always_tags,
                    base_tags=list(ctx.hashtags or []),
                    db_pool=db_pool,
                    strategy=strategy,
                )
                if meta.get("ok"):
                    used_m8 = True
                    tok = meta.get("tokens") or {}
                    logger.info(
                        "M8 caption engine: prompt=%s completion=%s category=%s",
                        tok.get("prompt", 0),
                        tok.get("completion", 0),
                        category,
                    )
                    if generate_hashtags:
                        logger.info(
                            "AI hashtags finalised (%s): %s",
                            len(ctx.ai_hashtags or []),
                            ctx.ai_hashtags,
                        )
                    m8c = getattr(ctx, "m8_platform_captions", None) or {}
                    m8t = getattr(ctx, "m8_platform_titles", None) or {}
                    _trace_caption(ctx, str(ctx.upload_id), "m8_engine_ok", {
                        "m8_platform_caption_keys": sorted(
                            str(k).lower() for k in (m8c.keys() if isinstance(m8c, dict) else [])
                        ),
                        "m8_platform_title_keys": sorted(
                            str(k).lower() for k in (m8t.keys() if isinstance(m8t, dict) else [])
                        ),
                        "target_platforms": [str(p).lower() for p in (ctx.platforms or [])],
                    })
                    if isinstance(m8c, dict) and ctx.platforms and not any(
                        str(m8c.get(str(p).lower()) or "").strip() for p in (ctx.platforms or [])
                    ):
                        try:
                            ctx.output_artifacts["m8_degraded_reason"] = json.dumps(
                                {"ok": True, "issue": "m8_empty_per_platform_captions"},
                                default=str,
                            )[:2000]
                        except Exception:
                            pass
                        _trace_caption(ctx, str(ctx.upload_id), "m8_degraded_per_platform", {
                            "issue": "m8_empty_per_platform_captions",
                            "legacy_fallback_next": False,
                            "target_platforms": [str(p).lower() for p in (ctx.platforms or [])],
                        })
                else:
                    logger.warning(
                        "M8 caption engine failed (%s); falling back to legacy narrative prompt",
                        meta.get("error"),
                    )
                    try:
                        ctx.output_artifacts["m8_degraded_reason"] = json.dumps(
                            {
                                "ok": False,
                                "error": meta.get("error"),
                                "legacy_fallback": True,
                            },
                            default=str,
                        )[:4000]
                    except Exception:
                        pass
                    _trace_caption(ctx, str(ctx.upload_id), "m8_engine_failed", {
                        "legacy_fallback": True,
                        "error": str(meta.get("error") or "")[:800],
                    })
            except Exception as m8_err:
                logger.warning(
                    "M8 caption path error: %s; falling back to legacy narrative prompt",
                    m8_err,
                )
                try:
                    ctx.output_artifacts["m8_degraded_reason"] = json.dumps(
                        {"ok": False, "error": str(m8_err)[:800], "legacy_fallback": True},
                        default=str,
                    )[:4000]
                except Exception:
                    pass
                _trace_caption(ctx, str(ctx.upload_id), "m8_engine_exception", {
                    "legacy_fallback": True,
                    "error": str(m8_err)[:800],
                })

        if not used_m8 and _USE_M8_CAPTION_ENGINE:
            raw_deg = ""
            try:
                raw_deg = str(
                    (ctx.output_artifacts or {}).get("m8_degraded_reason") or ""
                )
            except Exception:
                raw_deg = ""
            _trace_caption(ctx, str(ctx.upload_id), "m8_legacy_fallback", {
                "legacy_fallback": True,
                "m8_degraded_reason": raw_deg[:2000],
            })

        if not used_m8:
            # ── Legacy: single category-aware narrative prompt ───────────────
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
                category=category,
            )
            _trace_caption(ctx, str(ctx.upload_id), "legacy_prompt", {
                "category": category,
                "prompt_chars": len(prompt),
                "prompt_preview": prompt,
                "frame_count": len(frames),
            })

            result = await _call_openai(
                frames=frames,
                prompt=prompt,
                model=model,
                hashtag_count=hashtag_count,
            )

            titles_bp = result.get("titles_by_platform") if isinstance(result.get("titles_by_platform"), dict) else {}
            caps_bp = result.get("captions_by_platform") if isinstance(result.get("captions_by_platform"), dict) else {}
            tags_bp = result.get("hashtags_by_platform") if isinstance(result.get("hashtags_by_platform"), dict) else {}

            ctx.m8_platform_titles = dict(titles_bp)
            ctx.m8_platform_captions = dict(caps_bp)
            ctx.m8_platform_hashtags = {}
            for pl_h, raw_list in (tags_bp or {}).items():
                if not isinstance(raw_list, list):
                    continue
                ctx.m8_platform_hashtags[str(pl_h).lower()] = _finalise_hashtags(
                    ai_tags=raw_list,
                    base_tags=list(ctx.hashtags or []) + always_tags,
                    blocked=blocked_tags,
                    max_total=pref_max,
                )

            if result.get("title") and generate_title:
                ctx.ai_title = str(result["title"]).strip()[:120]
            elif generate_title and titles_bp.get("youtube"):
                ctx.ai_title = str(titles_bp["youtube"]).strip()[:120]
            elif generate_title and titles_bp:
                ctx.ai_title = str(next(iter(titles_bp.values()))).strip()[:120]

            if result.get("caption") and generate_caption:
                ctx.ai_caption = strip_stray_hashtag_json_blob(str(result["caption"]).strip())[:500]
            elif generate_caption and caps_bp.get("tiktok"):
                ctx.ai_caption = strip_stray_hashtag_json_blob(str(caps_bp["tiktok"]).strip())[:500]
            elif generate_caption:
                for _k in ("tiktok", "instagram", "facebook", "youtube"):
                    if caps_bp.get(_k):
                        ctx.ai_caption = strip_stray_hashtag_json_blob(str(caps_bp[_k]).strip())[:500]
                        break

            if generate_hashtags:
                if result.get("hashtags"):
                    ai_raw = result.get("hashtags") or []
                    ctx.ai_hashtags = _finalise_hashtags(
                        ai_tags=ai_raw,
                        base_tags=list(ctx.hashtags or []) + always_tags,
                        blocked=blocked_tags,
                        max_total=pref_max,
                    )
                elif tags_bp.get("tiktok"):
                    ctx.ai_hashtags = _finalise_hashtags(
                        ai_tags=tags_bp.get("tiktok") or [],
                        base_tags=list(ctx.hashtags or []) + always_tags,
                        blocked=blocked_tags,
                        max_total=pref_max,
                    )
                elif tags_bp.get("instagram"):
                    ctx.ai_hashtags = _finalise_hashtags(
                        ai_tags=tags_bp.get("instagram") or [],
                        base_tags=list(ctx.hashtags or []) + always_tags,
                        blocked=blocked_tags,
                        max_total=pref_max,
                    )
                elif tags_bp.get("facebook"):
                    ctx.ai_hashtags = _finalise_hashtags(
                        ai_tags=tags_bp.get("facebook") or [],
                        base_tags=list(ctx.hashtags or []) + always_tags,
                        blocked=blocked_tags,
                        max_total=pref_max,
                    )
                elif tags_bp.get("youtube"):
                    ctx.ai_hashtags = _finalise_hashtags(
                        ai_tags=tags_bp.get("youtube") or [],
                        base_tags=list(ctx.hashtags or []) + always_tags,
                        blocked=blocked_tags,
                        max_total=pref_max,
                    )
                logger.info(
                    f"AI hashtags finalised ({len(ctx.ai_hashtags or [])}): {ctx.ai_hashtags}"
                )

            if ctx.ai_title:
                logger.info(f"AI title: {ctx.ai_title[:80]}")
            if ctx.ai_caption:
                logger.info(f"AI caption: {ctx.ai_caption[:80]}")

            legacy_gap: Dict[str, Any] = {}
            try:
                plat_sel = [
                    str(p).strip().lower()
                    for p in (ctx.platforms or [])
                    if str(p).strip()
                ]
                plat_ok = frozenset({"youtube", "tiktok", "instagram", "facebook"})
                plat_sel = [p for p in plat_sel if p in plat_ok]
                if generate_caption and plat_sel:
                    missing_c = [
                        p for p in plat_sel if not str((caps_bp or {}).get(p) or "").strip()
                    ]
                    if missing_c:
                        legacy_gap["missing_caption_platforms"] = missing_c
                if generate_title and plat_sel:
                    missing_t = [
                        p for p in plat_sel if not str((titles_bp or {}).get(p) or "").strip()
                    ]
                    if missing_t:
                        legacy_gap["missing_title_platforms"] = missing_t
                if generate_hashtags and plat_sel:
                    missing_h = [p for p in plat_sel if not (tags_bp or {}).get(p)]
                    if missing_h:
                        legacy_gap["missing_hashtag_platforms"] = missing_h
                if legacy_gap:
                    ctx.output_artifacts["caption_legacy_per_platform_gap"] = json.dumps(
                        legacy_gap,
                        default=str,
                    )[:2000]
            except Exception:
                pass

            tokens_used = result.get("tokens", {})
            cost = _estimate_cost(tokens_used, len(frames))
            logger.info(
                f"OpenAI usage (legacy path): prompt={tokens_used.get('prompt', 0)}, "
                f"completion={tokens_used.get('completion', 0)}, "
                f"cost=${cost:.4f}, category={category}"
            )
            _trace_caption(ctx, str(ctx.upload_id), "legacy_result", {
                "title": str(ctx.ai_title or ""),
                "caption": str(ctx.ai_caption or ""),
                "hashtag_count": len(ctx.ai_hashtags or []),
                "titles_by_platform": dict(titles_bp),
                "captions_by_platform": {k: (str(v)[:220] + "…") if len(str(v)) > 220 else str(v) for k, v in caps_bp.items()},
                "hashtags_by_platform": {k: list(v)[:24] for k, v in (tags_bp or {}).items()},
                "legacy_per_platform_gap": legacy_gap,
                "tokens": tokens_used,
                "estimated_cost": round(cost, 6),
            })

        # ── Content attribution (settings snapshot → output_artifacts + DB) ──
        from core.content_attribution import (
            build_content_attribution_snapshot,
            collect_hashtag_slugs_for_attribution,
            content_attribution_strategy_key,
        )

        _ht_slugs = collect_hashtag_slugs_for_attribution(ctx)
        if ctx.ai_title or ctx.ai_caption or ctx.ai_hashtags or _ht_slugs:
            try:
                snap = build_content_attribution_snapshot(
                    user_settings=us,
                    strategy=strategy,
                    category=category,
                    used_m8_engine=used_m8,
                    caption_style_ui=caption_style,
                    caption_tone_ui=caption_tone,
                    caption_voice_ui=caption_voice,
                    hashtag_style=hashtag_style,
                    hashtag_count=hashtag_count if generate_hashtags else 0,
                    caption_frame_count=num_frames,
                    generate_hashtags=generate_hashtags,
                    output_artifacts=dict(ctx.output_artifacts or {}),
                    hashtag_slugs_used=_ht_slugs,
                )
                ctx.output_artifacts["content_attribution_v1"] = json.dumps(snap)
                ctx.output_artifacts["content_attribution_key"] = content_attribution_strategy_key(snap)
                if db_pool:
                    from . import db as _db_stage

                    await _db_stage.merge_job_output_artifacts_strings(
                        db_pool,
                        str(ctx.upload_id),
                        {
                            "content_attribution_v1": ctx.output_artifacts["content_attribution_v1"],
                            "content_attribution_key": ctx.output_artifacts["content_attribution_key"],
                        },
                    )
            except Exception as attr_err:
                logger.debug("content attribution: %s", attr_err)

        # ── Persist into caption memory for future few-shot retrieval ────────
        if db_pool and (ctx.ai_title or ctx.ai_caption or ctx.ai_hashtags):
            try:
                from . import db as _db_stage
                _voice = caption_voice
                await _db_stage.insert_caption_memory(
                    db_pool,
                    str(ctx.user_id),
                    str(ctx.upload_id),
                    category,
                    list(ctx.platforms or []),
                    ctx.ai_title,
                    ctx.ai_caption,
                    ctx.ai_hashtags,
                    caption_voice=_voice,
                    caption_tone=caption_tone,
                    caption_style=caption_style,
                )
            except Exception as _ins_err:
                logger.debug(f"Caption memory insert skipped: {_ins_err}")

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
