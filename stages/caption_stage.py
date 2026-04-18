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

from core.helpers import sanitize_hashtag_body

import httpx

from .errors import SkipStage, StageError, ErrorCode
from .context import JobContext
from .ai_service_costs import user_pref_ai_service_enabled

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

# M8_ENGINE path: scene graph + user-seeded content_strategy → OpenAI JSON → rank → ctx.
# Set UPLOADM8_M8_CAPTION_ENGINE=false to force the legacy single-prompt narrative path only.
_USE_M8_CAPTION_ENGINE = os.environ.get("UPLOADM8_M8_CAPTION_ENGINE", "true").lower() in (
    "1", "true", "yes", "on",
)

# Pricing per 1K tokens (gpt-4o-mini defaults; gpt-4o is ~10x)
COST_PER_1K_INPUT  = 0.000150
COST_PER_1K_OUTPUT = 0.000600
COST_PER_IMAGE     = 0.00765   # low-detail vision


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
            "High-energy, petrolhead. Use car culture vocabulary. "
            "Speed, freedom, and mechanical passion — make them feel the RPMs."
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


# Caption energy / register (user captionTone). Topic-agnostic: jargon and
# examples must always come from visible content + transcript, not from a
# assumed niche.
TONE_DIRECTIVES: Dict[str, str] = {
    "hype": (
        "High momentum and conviction: strong verbs, tight clauses, forward pull. "
        "You may use emphatic punctuation sparingly. "
        "Scale intensity to the subject—finance, grief, or slow crafts get "
        "'quiet hype' (urgent clarity) rather than party-bro shouting. "
        "Never invent stakes or drama not supported by the video."
    ),
    "calm": (
        "Measured, breathable pacing; let concrete details do the work. "
        "Prefer understatement over exclamation. "
        "Works for any subject: tutorials, newsy explainers, intimate vlogs, "
        "technical demos—same cool, trustworthy register."
    ),
    "cinematic": (
        "Scene-led, sensory language: light, shadow, motion, scale, texture—only "
        "what the frames support. Present tense where it heightens immediacy. "
        "Trailer-like rhythm without melodrama or clichés that could apply to "
        "any video; every line should tether to something visible or said."
    ),
    "authentic": (
        "Human, direct, first-person or close second-person; plain words over "
        "marketing speak. "
        "Sound like a real person in any niche—parenting, code, sports, "
        "small business—without filler ('okay guys', 'here's the thing' spam). "
        "One honest observation beats a generic hook."
    ),
}

# High-level speaker persona (user captionVoice). Static prose instructions only.
# Must adapt vocabulary to the actual topic; persona is delivery, not a fake niche.
VOICE_PROFILES: Dict[str, str] = {
    "default": (
        "Balanced creator voice: clear hook, specific middle, satisfying close. "
        "Confident but not performative; match slang and terminology to what "
        "the content actually is (chef terms for food, dev terms for code, etc.)."
    ),
    "mentor": (
        "Experienced guide: 'you'-oriented, encouraging, zero condescension. "
        "Implied expertise through specifics, not credentials flex. "
        "End with a usable takeaway when the video teaches or demonstrates something."
    ),
    "hypebeast": (
        "Peak short-form energy: clipped sentences, rhythm, occasional bold word "
        "choice—still believable. "
        "Slang only if it fits the subject and platform; avoid empty viral filler. "
        "Excitement must trace back to what is literally on screen or in audio."
    ),
    "best_friend": (
        "Warm, unfiltered peer: conversational fragments OK, light humor if "
        "the content allows. "
        "Self-aware without derailing; relatable across any hobby or life slice. "
        "Never mean-spirited or faux-chaos."
    ),
    "teacher": (
        "Educator clarity: one central idea, logical mini-arc, minimal jargon "
        "unless the audience clearly expects it from the visuals. "
        "If the clip is not instructional, still be precise—teach what "
        "happened or what to notice, not a life lesson unrelated to the footage."
    ),
    "cinematic_narrator": (
        "Third-person or omniscient trailer voice: declarative, image-stacking, "
        "slightly elevated register. "
        "Still anchored to real events in the clip—no epic narration of "
        "nothing happening. Save flourish for genuine peaks in the footage."
    ),
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
            f'NEVER return single letters or word fragments.'
        )
    tasks_block = "\n".join(tasks) if tasks else "(none requested)"

    # ── Tone + voice (user prefs): composable layers on top of category context
    tone_instruction = TONE_DIRECTIVES.get(caption_tone) or TONE_DIRECTIVES["authentic"]

    us = getattr(ctx, "user_settings", None) or {}
    voice_key = str(us.get("captionVoice") or us.get("caption_voice") or "default").lower()
    if voice_key not in VOICE_PROFILES:
        voice_key = "default"
    voice_instruction = VOICE_PROFILES[voice_key]

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
    context_lines = [f"Filename: {ctx.filename}", f"Target platforms: {platform_str}"]
    if ctx.title:
        context_lines.append(f"User title hint: {ctx.title}")
    if ctx.caption:
        context_lines.append(f"User caption hint: {ctx.caption}")
    if ctx.location_name:
        context_lines.append(f"Filming location: {ctx.location_name}")
    context_block = "\n".join(f"• {ln}" for ln in context_lines)

    # ── Memory examples (few-shot from past uploads) ─────────────────────────
    memory_block = ""
    examples = getattr(ctx, "caption_memory_examples", None) or []
    if examples:
        lines = []
        for i, ex in enumerate(examples[:5], 1):
            t = (ex.get("ai_title") or "")[:120]
            c = (ex.get("ai_caption") or "")[:320]
            tags = ex.get("ai_hashtags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            if isinstance(tags, list):
                tag_str = ", ".join(str(x).lstrip("#") for x in tags[:15])
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
        ocr = (vc.get("ocr_text") or "").strip()
        if ocr:
            bits.append(f"On-screen / OCR text: {ocr[:900]}")
        labels = vc.get("label_names") or []
        if labels:
            bits.append("Scene labels: " + ", ".join(str(x) for x in labels[:22]))
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
                "\n━━ FRAME INSPECTOR (Google Vision — sampled frame) ━━\n"
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
        summary = (vi_ctx.get("summary_text") or "").strip()
        if summary:
            video_intel_block = (
                f"\n━━ VIDEO ANALYZER (full clip — labels) ━━\n{summary[:2200]}\n"
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
        yev = ac.get("yamnet_events")
        if isinstance(yev, list) and yev:
            audio_enrich_block += (
                "\n━━ AUDIO EVENTS ━━\n"
                + ", ".join(str(x) for x in yev[:16])
                + "\n━━━━━━━━━━━━━━━━━\n"
            )

    # ── Trill beat ───────────────────────────────────────────────────────────
    trill_beat = _build_trill_beat(ctx)
    trill_section = f"\n\n{trill_beat}" if trill_beat else ""

    # ── Platform-specific output notes ───────────────────────────────────────
    platform_notes = []
    if "youtube" in (ctx.platforms or []):
        platform_notes.append("YouTube: provide both title and description.")
    if any(p in (ctx.platforms or []) for p in ("tiktok", "instagram", "facebook")):
        platform_notes.append("TikTok/Instagram/Facebook: caption only — no title field.")
    platform_note = " | ".join(platform_notes)

    prompt = f"""You are a social media content specialist for {platform_str}.

{frame_instruction}
{memory_block}
{category_block}{transcript_block}{vision_block}{scene_understanding_block}{video_intel_block}{audio_enrich_block}{trill_section}

TONE DIRECTIVE ({caption_tone.upper()}): {tone_instruction}
VOICE PROFILE ({voice_key.upper()}): {voice_instruction}
CAPTION STYLE: {caption_style.upper()} — follow this style strictly.
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
- Content must feel AUTHENTIC — not AI-generated
- NEVER invent events, locations, or narratives not supported by the frames, transcript, or telemetry provided. If no frames: do not claim specific visuals.
- ACCURACY OVER ENGAGEMENT: Do NOT use clickbait patterns ("Nobody expected", "You need to see this", "The secret nobody tells you"). Describe what is actually shown. Hooks must reflect visible content — never overpromise or mislead.
- Hook in the first 3 words for short-form platforms
- Use emojis sparingly (1–3 max)
- HASHTAGS: each must be a complete word (e.g. "makeuptutorial", "gardenlife", "dashcam")
  NEVER return single characters or word fragments
- Be SPECIFIC to what is actually visible — generic content gets buried
- If Trill data provided: caption MUST reference at least one real data point

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
    """Call OpenAI with prompt + attached frames. Returns {title, caption, hashtags, tokens}."""
    result: Dict[str, Any] = {
        "title": None, "caption": None, "hashtags": [], "tokens": {}
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
    if not can_ai:
        raise SkipStage("AI not available on this plan")

    us = ctx.user_settings or {}

    if not user_pref_ai_service_enabled(us, "caption_llm", default=True):
        raise SkipStage("Caption writer disabled (aiServiceCaptionWriter=false)")

    # ── User preference toggles ──────────────────────────────────────────────
    generate_caption  = bool(us.get("autoCaptions") or us.get("auto_captions") or False)
    generate_title    = generate_caption
    generate_hashtags = bool(us.get("aiHashtagsEnabled") or us.get("ai_hashtags_enabled") or False)

    if us.get("auto_generate_captions") is not None:
        generate_caption = bool(us["auto_generate_captions"])
        generate_title   = generate_caption
    if us.get("auto_generate_hashtags") is not None:
        generate_hashtags = bool(us["auto_generate_hashtags"])

    # Upload-page title/caption are authoritative: do not spend tokens replacing them.
    if (ctx.title or "").strip():
        generate_title = False
    if (ctx.caption or "").strip():
        generate_caption = False

    if not (generate_title or generate_caption or generate_hashtags):
        raise SkipStage("AI generation not enabled by user settings")

    # ── Style/tone preferences ───────────────────────────────────────────────
    caption_style = str(us.get("captionStyle") or us.get("caption_style") or "story").lower()
    if caption_style not in ("story", "punchy", "factual"):
        caption_style = "story"

    caption_tone = str(us.get("captionTone") or us.get("caption_tone") or "authentic").lower()
    if caption_tone not in ("hype", "calm", "cinematic", "authentic"):
        caption_tone = "authentic"

    caption_voice = str(us.get("captionVoice") or us.get("caption_voice") or "default").lower()
    if caption_voice not in (
        "default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator",
    ):
        caption_voice = "default"

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
    if not user_pref_ai_service_enabled(us, "vision_google", default=True):
        num_frames = 0

    # ── Detect content category (3-layer: hint → filename → general) ─────────
    category = _detect_content_category(ctx)

    # ── Retrieve few-shot examples from upload_caption_memory (optional) ───
    if db_pool:
        try:
            from . import db as _db_stage
            ctx.caption_memory_examples = await _db_stage.fetch_caption_memory_examples(
                db_pool, str(ctx.user_id), category, limit=3
            )
        except Exception as _mem_err:
            logger.debug(f"Caption memory fetch skipped: {_mem_err}")

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
                else:
                    logger.warning(
                        "M8 caption engine failed (%s); falling back to legacy narrative prompt",
                        meta.get("error"),
                    )
            except Exception as m8_err:
                logger.warning(
                    "M8 caption path error: %s; falling back to legacy narrative prompt",
                    m8_err,
                )

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

            result = await _call_openai(
                frames=frames,
                prompt=prompt,
                model=model,
                hashtag_count=hashtag_count,
            )

            if result.get("title") and generate_title:
                ctx.ai_title = result["title"]
                logger.info(f"AI title: {ctx.ai_title[:80]}")

            if result.get("caption") and generate_caption:
                ctx.ai_caption = result["caption"]
                logger.info(f"AI caption: {ctx.ai_caption[:80]}")

            if generate_hashtags:
                ai_raw = result.get("hashtags") or []
                ctx.ai_hashtags = _finalise_hashtags(
                    ai_tags=ai_raw,
                    base_tags=list(ctx.hashtags or []) + always_tags,
                    blocked=blocked_tags,
                    max_total=pref_max,
                )
                logger.info(
                    f"AI hashtags finalised ({len(ctx.ai_hashtags)}): {ctx.ai_hashtags}"
                )

            tokens_used = result.get("tokens", {})
            cost = _estimate_cost(tokens_used, len(frames))
            logger.info(
                f"OpenAI usage (legacy path): prompt={tokens_used.get('prompt', 0)}, "
                f"completion={tokens_used.get('completion', 0)}, "
                f"cost=${cost:.4f}, category={category}"
            )

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
