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

import httpx

from .errors import SkipStage, StageError, ErrorCode
from .context import JobContext

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

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
            "The secret to glowy skin nobody tells you 🌟",
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
            "Nobody believed this would work. I proved them wrong.",
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
            "Nobody expects this strat in ranked 👀",
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
            "Nobody teaches this in school — here's what you need to know",
            "I learned this the hard way so you don't have to",
            "The 1% of people who know this will get ahead",
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
            "This house generates ${revenue}/month passively",
            "I bought my first rental property at 24 — here's how",
            "The neighbourhood nobody is talking about yet",
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
            "Nobody thought we'd make it to the finals",
            "The play that silenced the critics",
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
            "The most satisfying 60 seconds you'll watch today",
            "Just watch until the end 🎧",
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
            "I tried {challenge} for 30 days — here's what changed",
            "Morning routine that actually changed my life",
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
            "Identify the content type from the frames and match your tone to it."
        ),
        "hook_templates": [
            "You need to see this",
            "Nobody expected this outcome",
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
        lines.append(f"HOOK INSPIRATION (adapt freely, never copy verbatim): {hook_examples}")
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
        "captions get algorithmic lift. Hook must feel native to this content type. ━━"
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

    # ── Tone: user pref overrides category default ───────────────────────────
    tone_instruction = {
        "hype":      "Energy is HIGH. Power words, exclamation, urgency. Stop the scroll.",
        "calm":      "Measured and confident. Let the footage speak. Understated cool.",
        "cinematic": "Poetic, atmospheric. Paint a picture with words. Film trailer voiceover.",
        "authentic": "Real talk, first-person, no fluff. Like texting a friend.",
    }.get(caption_tone) or CONTENT_CATEGORIES.get(
        category, CONTENT_CATEGORIES["general"]
    ).get("tone", "Engaging and authentic.")

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
    else:
        frame_instruction = (
            "You are being shown 1 frame. "
            "Identify everything visible: environment, subject, activity, products, energy, "
            "branding, skill level. Use all visible signals for specific content."
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

    # ── Category context block ────────────────────────────────────────────────
    category_block = _build_category_context_block(category, ctx.location_name)

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

{category_block}{trill_section}

TONE DIRECTIVE: {tone_instruction}
CAPTION STYLE: {caption_style.upper()} — follow this style strictly.
{f"PLATFORM NOTE: {platform_note}" if platform_note else ""}

Generate the following for this video:
{tasks_block}

Context:
{context_block}

Rules:
- Content must feel AUTHENTIC — not AI-generated
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
                    tag = str(tag).strip().lstrip("#").lower()
                    if len(tag) < 2:
                        continue
                    if " " in tag:
                        for part in tag.split():
                            p = part.lstrip("#").lower()
                            if len(p) >= 2:
                                cleaned.append(p)
                    else:
                        cleaned.append(tag)

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

    # ── User preference toggles ──────────────────────────────────────────────
    generate_caption  = bool(us.get("autoCaptions") or us.get("auto_captions") or False)
    generate_title    = generate_caption
    generate_hashtags = bool(us.get("aiHashtagsEnabled") or us.get("ai_hashtags_enabled") or False)

    if us.get("auto_generate_captions") is not None:
        generate_caption = bool(us["auto_generate_captions"])
        generate_title   = generate_caption
    if us.get("auto_generate_hashtags") is not None:
        generate_hashtags = bool(us["auto_generate_hashtags"])

    if not (generate_title or generate_caption or generate_hashtags):
        raise SkipStage("AI generation not enabled by user settings")

    # ── Style/tone preferences ───────────────────────────────────────────────
    caption_style = str(us.get("captionStyle") or us.get("caption_style") or "story").lower()
    if caption_style not in ("story", "punchy", "factual"):
        caption_style = "story"

    caption_tone = str(us.get("captionTone") or us.get("caption_tone") or "authentic").lower()
    if caption_tone not in ("hype", "calm", "cinematic", "authentic"):
        caption_tone = "authentic"

    hashtag_style = str(us.get("aiHashtagStyle") or us.get("ai_hashtag_style") or "mixed").lower()
    if hashtag_style not in ("trending", "niche", "mixed"):
        hashtag_style = "mixed"

    pref_max = int(
        us.get("aiHashtagCount") or us.get("ai_hashtag_count") or
        us.get("maxHashtags") or us.get("max_hashtags") or 15
    )
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
    category = _detect_content_category(ctx)

    logger.info(
        f"Caption stage: category={category}, style={caption_style}, tone={caption_tone}, "
        f"hashtag_style={hashtag_style}, count={hashtag_count}, "
        f"model={model}, frames={num_frames}"
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
        # ── Collect story frames ─────────────────────────────────────────────
        frames = await _collect_story_frames(ctx, num_frames)
        if not frames:
            logger.warning("No frames available — generating captions without visual context")

        # ── Build prompt (category-aware) ─────────────────────────────────────
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

        if generate_hashtags:
            ai_raw = result.get("hashtags") or []

            # ── FIX ISSUE 3: _finalise_hashtags() IS NOW CALLED ──────────────
            # Previously this function existed but was never invoked, allowing
            # blocked hashtags to appear in every published post.
            # Now: blocked tags filtered out, always_hashtags merged in first.
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
            f"OpenAI usage: prompt={tokens_used.get('prompt', 0)}, "
            f"completion={tokens_used.get('completion', 0)}, "
            f"cost=${cost:.4f}, category={category}"
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
