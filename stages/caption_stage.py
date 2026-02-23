"""
UploadM8 Caption Stage — Human Voice Engine
=============================================

The old stage used a generic "be engaging and authentic" prompt.
That is the fastest way to get AI-sounding output.

This rebuild uses a completely different approach:
  1. PERSONA-BASED PROMPTING — GPT plays a specific character (a real car
     person who posts on socials), not a "content creation expert"
  2. FEW-SHOT EXAMPLES — 3-5 real-sounding posts shown per trill bucket
     so GPT calibrates tone from examples, not instructions
  3. TRILL BUCKET VOICE ENGINE — each bucket (gloryBoy → chill) gets its
     own emotional register, vocabulary, and sentence rhythm
  4. NATURAL TELEMETRY TRANSLATION — pandas stats (p95, accel_events,
     speeding_seconds) become phrases like "stayed in the 120s for a good
     8 seconds" rather than labeled data tables
  5. PLATFORM MICRO-RULES — TikTok gets lowercase POV energy, YouTube
     gets slightly more description, Instagram gets visual/aesthetic,
     Facebook gets broader/conversational
  6. ANTI-CLICHE BLOCKLIST — explicit list of AI giveaway phrases that
     are forbidden in every single prompt (adrenaline, buckle up, etc.)
  7. HIGH TEMPERATURE + RANDOMIZED EXAMPLES — variance so 500 uploads
     don't all sound identical

PANDAS FLOW:
  telemetry_stage.py computes speed_p95, accel_events, speeding_seconds,
  euphoria_seconds via DataFrame operations and stores them as dynamic
  attrs on TelemetryData. This stage reads them and translates to human
  language before injecting into prompts.
"""

import asyncio
import base64
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .context import JobContext, TelemetryData, TrillScore
from .errors import SkipStage, StageError, ErrorCode

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")

COST_PER_1K_INPUT = 0.0025
COST_PER_1K_OUTPUT = 0.010
COST_PER_IMAGE = 0.000213  # low detail, gpt-4o

# ─────────────────────────────────────────────────────────────────────────────
# VOICE LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

# These phrases IMMEDIATELY flag content as AI-generated.
# They're banned from every prompt without exception.
AI_CLICHES = [
    "buckle up", "heart pumping", "adrenaline", "adrenaline rush",
    "epic journey", "thrilling experience", "pushing the limits",
    "content creator", "I wanted to share", "in this video",
    "join me as", "don't forget to", "hit that like button",
    "incredible journey", "pure excitement", "heart-pounding",
    "breathtaking", "exhilarating", "rev up", "feel the rush",
    "living my best life", "blessed and grateful", "making memories",
    "road warrior", "passion for driving", "the open road",
    "need for speed", "born to drive", "driving enthusiast",
    "car enthusiast community", "fellow enthusiasts",
    "I am so excited", "absolutely incredible", "truly amazing",
    "the thrill of", "the joy of", "unforgettable experience",
    "I hope you enjoy", "don't forget to like", "comment below",
    "stay tuned", "as always", "without further ado",
]

# Per-bucket emotional register — tells GPT WHO they are right now
BUCKET_VOICE: Dict[str, str] = {
    "gloryBoy": (
        "You just hit something insane and you're still processing it. "
        "Short bursts. Slightly unhinged. You're not bragging — you're "
        "trying to describe something that barely felt real. "
        "Energy is 11/10. Sentences are short. Some incomplete. "
        "This is the kind of thing you send to your car group chat at midnight."
    ),
    "euphoric": (
        "That was a clean, elite run and you know it. "
        "Confident without being obnoxious. You're stating facts, not showing off. "
        "The kind of post you tap out while you're still a little buzzed from the drive. "
        "Not hype, just settled confidence."
    ),
    "sendIt": (
        "Solid committed run, you're pumped. "
        "Casual energy but you're clearly proud. "
        "Like when you're telling the story at a meet and people actually lean in. "
        "Medium length sentences, genuine enthusiasm."
    ),
    "spirited": (
        "Good clean drive. You're happy about it. "
        "The kind of caption you dash off quickly because the clip came out nice. "
        "Relaxed, real. No performance."
    ),
    "chill": (
        "Just vibing. Good drive, decent clip. "
        "Zero try-hard energy. Easy casual tone. "
        "Like a Tuesday for you. Not every clip needs to be a highlight."
    ),
}

# Platform micro-rules — stacked ON TOP of bucket voice
PLATFORM_RULES: Dict[str, str] = {
    "tiktok": (
        "Platform: TikTok. "
        "First 2 words are your hook — they have to grab in the feed. "
        "Lowercase usually lands better. Short punchy sentences. "
        "POV: hooks work well for dashcam footage. 1-3 sentences MAX. "
        "No period at the end of the last sentence. Feels like a text, not a post."
    ),
    "youtube": (
        "Platform: YouTube Shorts. "
        "Slightly more descriptive than TikTok — 1-2 sentences. "
        "Normal capitalization. Mention what makes this clip worth watching. "
        "Still casual. Not a YouTube essay, just a short punchy description."
    ),
    "instagram": (
        "Platform: Instagram Reels. "
        "Visual-first energy. 2-3 sentences. "
        "Reference what you're seeing or feeling in the clip. "
        "Can be slightly more polished but never curated. "
        "Like a caption on a real car account, not a brand page."
    ),
    "facebook": (
        "Platform: Facebook Reels. "
        "3-4 sentences OK. Slightly more conversational, broader audience. "
        "Can reference location more directly. "
        "Normal capitalization. Relatable tone."
    ),
}

# Few-shot examples per bucket — GPT calibrates FROM these, not from instructions
FEW_SHOT: Dict[str, Dict[str, List[str]]] = {
    "gloryBoy": {
        "titles": [
            "bro what just happened",
            "the data does not lie",
            "okay that got out of hand",
            "i need a minute",
            "the map file said what",
            "we cooked",
            "that one got away from me",
        ],
        "captions": [
            "wasn't expecting that. pulled the data after and had to scroll twice to believe it.",
            "three runs back and i still can't fully explain that one. the numbers though.",
            "everything lined up at exactly the wrong time. or the right time. depends who you ask.",
            "the car decided tonight was the night. i was just along for it.",
            "i'll let the data speak. some things don't need a caption.",
            "sat in the parking lot for five minutes after this one. just staring at the map.",
            "some nights the road just hands you something. this was one of those.",
        ],
    },
    "euphoric": {
        "titles": [
            "she woke up today",
            "conditions were right",
            "peak form right here",
            "this is what it's built for",
            "clean all the way through",
            "the car was dialed",
            "no complaints",
        ],
        "captions": [
            "everything clicked. grip was there, traffic cleared, hands just knew what to do.",
            "some clips you watch back and just nod. this is one of those.",
            "been chasing a run like this for a while. finally got it on camera.",
            "car felt alive tonight. you can hear it in the audio.",
            "when the gap opens and you're already committed. no second thoughts.",
            "not every run comes together like this. glad the camera was rolling.",
            "clean entry, clean exit. that's really all you can ask for.",
        ],
    },
    "sendIt": {
        "titles": [
            "committed to it",
            "no hesitation",
            "full send",
            "let her breathe",
            "clean exit",
            "rolled the dice",
            "gap was there",
        ],
        "captions": [
            "left some room but not that much. worked out.",
            "wasn't sure about the gap until i was already in it. that's usually how it goes.",
            "car wanted to go and i wasn't about to argue with it.",
            "sometimes you just send it and figure the rest out on the way.",
            "solid run. not my best but honestly no complaints.",
            "the gap looked bigger in my head. still made it work.",
            "committed before i thought about it too much. right call.",
        ],
    },
    "spirited": {
        "titles": [
            "good morning",
            "that'll do",
            "casual tuesday energy",
            "just a drive",
            "the commute actually slapped",
            "not bad at all",
            "clean run nothing crazy",
        ],
        "captions": [
            "nothing crazy just vibing. clip came out nice.",
            "wasn't even trying. sometimes those are the best ones.",
            "clean roads, decent weather, no complaints from anyone in the car.",
            "just getting home but make it content.",
            "the car doesn't care what day of the week it is.",
            "easy drive. sometimes that's exactly what you need.",
            "good run, good clip, going home.",
        ],
    },
    "chill": {
        "titles": [
            "just a drive",
            "nothing crazy",
            "the usual",
            "good roads today",
            "evening cruise",
            "out here",
            "rolling",
        ],
        "captions": [
            "sometimes the best drives are the ones you almost didn't take.",
            "roads were clear. that's really all it takes.",
            "no destination, just moving. needed it.",
            "car sounds different when you're not in a rush.",
            "not every clip needs to be a highlight reel and that's fine.",
            "just out here. nothing more to report.",
            "easy one tonight. exactly what i needed.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

async def _video_duration(path: Path) -> float:
    try:
        cmd = [FFPROBE_PATH, "-v", "quiet", "-show_entries", "format=duration", "-of", "json", str(path)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        return float(json.loads(out.decode()).get("format", {}).get("duration", 30.0))
    except Exception:
        return 30.0


async def _extract_frame(path: Path, out: Path, ts: float) -> bool:
    cmd = [
        FFMPEG_PATH, "-y", "-ss", f"{ts:.3f}", "-i", str(path),
        "-vframes", "1", "-q:v", "4", "-vf", "scale=720:-2", str(out)
    ]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return out.exists() and out.stat().st_size > 1000
    except Exception:
        return False


async def extract_frames(video_path: Path, temp_dir: Path, n: int = 8) -> List[Path]:
    if not video_path or not video_path.exists():
        return []
    dur = await _video_duration(video_path)
    start = dur * 0.05
    end = dur * 0.95
    span = end - start
    if span <= 0:
        span = dur
        start = 0
    interval = span / max(n, 1)
    timestamps = [start + interval * i + interval / 2 for i in range(n)]
    outs = [temp_dir / f"cf_{i:03d}.jpg" for i in range(n)]
    results = await asyncio.gather(*[_extract_frame(video_path, outs[i], ts) for i, ts in enumerate(timestamps)], return_exceptions=True)
    frames = [outs[i] for i, ok in enumerate(results) if ok is True and outs[i].exists()]
    logger.info(f"Extracted {len(frames)}/{n} frames")
    return frames


def frames_to_b64(frames: List[Path], max_frames: int = 6) -> List[str]:
    out = []
    for f in frames[:max_frames]:
        try:
            data = f.read_bytes()
            if len(data) > 500:
                out.append(base64.b64encode(data).decode())
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY → NATURAL LANGUAGE
# Converts pandas DataFrame stats into how a real driver would describe a run
# ─────────────────────────────────────────────────────────────────────────────

def _safe_attr(obj: Any, *names: str, default=None) -> Any:
    """Read first matching attribute name, return default if none found."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def telem_to_prose(ctx: JobContext) -> Dict[str, Any]:
    """
    Translate raw telemetry numbers into natural language fragments.

    Reads from TelemetryData (including pandas-enriched attrs set by
    telemetry_stage: _speed_p95, _accel_events, _speed_std) and
    TrillScore to build phrases a real driver would actually say.

    Returns a plain dict used by prompt builders below.
    """
    telem: Optional[TelemetryData] = ctx.telemetry or ctx.telemetry_data
    trill: Optional[TrillScore] = ctx.trill or ctx.trill_score

    result = {
        "location": None,         # "Kansas City, MO"
        "road": None,             # "I-70"
        "location_line": None,    # full prose location line
        "speed_phrase": None,     # "touching 118"
        "sustained_line": None,   # "held it for about 12 seconds"
        "accel_note": None,       # "a couple hard pulls"
        "distance_note": None,    # "3.2 mile run"
        "trill_bucket": "chill",
        "trill_score": None,
        "excessive": False,
    }

    if trill:
        result["trill_bucket"] = trill.bucket or "chill"
        result["trill_score"] = trill.score if trill.score else None
        result["excessive"] = bool(trill.excessive_speed)

    if not telem:
        return result

    # ── Location (from Nominatim reverse geocode in telemetry_stage) ─────────
    city = telem.location_city
    state = telem.location_state
    display = telem.location_display
    road = telem.location_road

    if display:
        result["location"] = display
    elif city and state:
        result["location"] = f"{city}, {state}"
    elif city:
        result["location"] = city

    if road:
        result["road"] = road

    if result["location"] and road:
        result["location_line"] = f"filmed in {result['location']} on {road}"
    elif result["location"]:
        result["location_line"] = f"filmed in {result['location']}"
    elif road:
        result["location_line"] = f"filmed on {road}"

    # ── Speed (pandas p95, max, speeding/euphoria seconds) ───────────────────
    max_spd = telem.max_speed_mph
    p95 = _safe_attr(telem, "_speed_p95", default=None)
    speeding_s = telem.speeding_seconds or 0.0
    euphoria_s = telem.euphoria_seconds or 0.0

    # Use p95 for the "typical top speed" feel, max for the peak
    top_ref = float(p95) if p95 else float(max_spd)

    if max_spd >= 130:
        result["speed_phrase"] = f"well into the 130s"
    elif max_spd >= 120:
        result["speed_phrase"] = f"touching {int(max_spd)}"
    elif max_spd >= 110:
        result["speed_phrase"] = f"into the 110s"
    elif max_spd >= 100:
        result["speed_phrase"] = "triple digits"
    elif max_spd >= 90:
        result["speed_phrase"] = f"pushing into the 90s"
    elif max_spd >= 80:
        result["speed_phrase"] = f"sitting in the high 80s"
    elif max_spd >= 70:
        result["speed_phrase"] = f"cruising 70-something"
    else:
        result["speed_phrase"] = "keeping it reasonable"

    # Sustained speed line (pandas speeding/euphoria seconds)
    if euphoria_s >= 10:
        result["sustained_line"] = f"held it for about {int(euphoria_s)} seconds"
    elif euphoria_s >= 5:
        result["sustained_line"] = f"briefly at that level"
    elif speeding_s >= 20:
        result["sustained_line"] = f"stayed up there for {int(speeding_s)} seconds"
    elif speeding_s >= 8:
        result["sustained_line"] = f"had a good {int(speeding_s)}-second run at that speed"

    # ── Acceleration events (pandas accel_events = delta speed > 10 mph) ─────
    accel = _safe_attr(telem, "_accel_events", default=0)
    accel = int(accel) if accel else 0
    if accel >= 6:
        result["accel_note"] = "multiple hard pulls throughout"
    elif accel >= 3:
        result["accel_note"] = "a few hard pulls in there"
    elif accel == 2:
        result["accel_note"] = "a couple hard pulls"
    elif accel == 1:
        result["accel_note"] = "one clean pull"

    # ── Distance (from pandas integration) ───────────────────────────────────
    dist = telem.total_distance_miles or 0.0
    if dist >= 15:
        result["distance_note"] = f"{dist:.0f} mile run"
    elif dist >= 5:
        result["distance_note"] = f"{dist:.1f} miles of this"
    elif dist >= 2:
        result["distance_note"] = f"just over {dist:.1f} miles"
    elif dist >= 0.5:
        result["distance_note"] = f"{dist:.1f} mile clip"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_system(bucket: str, platform: str, mode: str) -> str:
    """
    Build the system prompt using the persona + bucket + platform stack.
    mode = "title" | "caption" | "hashtags"
    """
    voice = BUCKET_VOICE.get(bucket, BUCKET_VOICE["chill"])
    plat_rule = PLATFORM_RULES.get(platform.lower(), PLATFORM_RULES["tiktok"])
    examples = FEW_SHOT.get(bucket, FEW_SHOT["chill"])

    # Pick 3 random examples for variety across 500 uploads
    if mode == "title":
        sampled = random.sample(examples["titles"], min(3, len(examples["titles"])))
        ex_block = "\n".join(f'  "{e}"' for e in sampled)
        task_line = (
            "Write ONE title. Max 80 characters. "
            "Raw text only — no quotes around it, no label, no explanation. "
            "Just the title on a single line.\n\n"
            f"Reference energy (do NOT copy these exactly, use them to calibrate tone):\n{ex_block}"
        )
    elif mode == "caption":
        sampled = random.sample(examples["captions"], min(3, len(examples["captions"])))
        ex_block = "\n".join(f'  "{e}"' for e in sampled)
        task_line = (
            "Write ONE caption. 1-3 sentences. "
            "Raw text only — no quotes, no label, no explanation.\n\n"
            f"Reference energy:\n{ex_block}"
        )
    else:  # hashtags
        task_line = (
            "Return ONLY a JSON array of hashtag strings starting with #. "
            "Nothing else. No markdown, no explanation. "
            'Example: ["#fyp", "#carsoftiktok", "#dashcam"]'
        )

    # Random sample of cliches to block (keeps prompt tight)
    blocked = ", ".join(f'"{w}"' for w in random.sample(AI_CLICHES, min(10, len(AI_CLICHES))))

    return f"""You are a car person who posts dashcam clips on social media. Not a "content creator." Not a marketer. Just someone who drives and posts about it.

Your energy right now: {voice}

{plat_rule}

HARD RULES — violating these makes the output unusable:
- NEVER use any of these phrases: {blocked}
- NEVER use em dashes (—)
- NEVER start a sentence with "I just" or "Just wanted to"  
- NEVER explain what the video shows — let the viewer watch it
- NEVER use the word "journey" "adventure" "experience" "thrill" or "adrenaline"
- NEVER sound like you're marketing to someone
- Write like a real person texted this, not like a brand posted it

{task_line}"""


def _build_user_prompt(prose: Dict[str, Any], has_frames: bool, ctx: JobContext) -> str:
    """
    Build the user-facing prompt with natural telemetry context.
    Data flows in as how a driver would think about a run.
    """
    parts = []

    if prose["location_line"]:
        parts.append(f"This clip was {prose['location_line']}.")

    if prose["speed_phrase"]:
        speed_line = f"Speed hit {prose['speed_phrase']}"
        if prose["sustained_line"]:
            speed_line += f" — {prose['sustained_line']}"
        speed_line += "."
        parts.append(speed_line)

    if prose["accel_note"]:
        parts.append(f"There were {prose['accel_note']}.")

    if prose["distance_note"]:
        parts.append(f"This was a {prose['distance_note']}.")

    if prose["trill_score"] and prose["trill_bucket"] in ("gloryBoy", "euphoric"):
        parts.append(f"Trill score: {prose['trill_score']}/100.")

    if has_frames:
        parts.append("Dashcam frames are attached. Use what you can actually see in them.")

    if ctx.title and ctx.title.strip():
        parts.append(f"Creator's own note: \"{ctx.title.strip()}\"")
    elif ctx.caption and ctx.caption.strip():
        parts.append(f"Creator's note: \"{ctx.caption.strip()}\"")

    if not parts and ctx.filename:
        parts.append(f"Filename: {ctx.filename}")

    return " ".join(parts) if parts else "Dashcam clip, no additional context."


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI CALL
# ─────────────────────────────────────────────────────────────────────────────

async def _openai(
    system: str,
    user: str,
    max_tokens: int = 150,
    images_b64: Optional[List[str]] = None,
    temperature: float = 0.92,
) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    model = OPENAI_VISION_MODEL if images_b64 else OPENAI_TEXT_MODEL
    try:
        if images_b64:
            content: Any = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}} for b64 in images_b64]
            content.append({"type": "text", "text": user})
        else:
            content = user

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
        if r.status_code == 429:
            logger.warning("OpenAI rate limited — skipping")
            return None
        if r.status_code != 200:
            logger.warning(f"OpenAI {r.status_code}: {r.text[:200]}")
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"OpenAI call error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

async def gen_title(ctx: JobContext, prose: Dict, images: List[str]) -> Optional[str]:
    platform = (ctx.platforms or ["tiktok"])[0].lower()
    system = _build_system(prose["trill_bucket"], platform, "title")
    user = _build_user_prompt(prose, bool(images), ctx) + "\n\nWrite the title now."
    raw = await _openai(system, user, max_tokens=60, images_b64=images or None, temperature=0.93)
    if not raw:
        return None
    # Strip any self-labeling GPT sometimes adds
    clean = raw.strip().strip('"').strip("'")
    for prefix in ("Title:", "title:", "TITLE:", "**Title:**"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
    return clean[:100]


async def gen_caption(ctx: JobContext, prose: Dict, images: List[str]) -> Optional[str]:
    platform = (ctx.platforms or ["tiktok"])[0].lower()
    system = _build_system(prose["trill_bucket"], platform, "caption")
    user = _build_user_prompt(prose, bool(images), ctx) + "\n\nWrite the caption now."
    raw = await _openai(system, user, max_tokens=220, images_b64=images or None, temperature=0.90)
    if not raw:
        return None
    clean = raw.strip().strip('"').strip("'")
    for prefix in ("Caption:", "caption:", "CAPTION:", "**Caption:**"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
    return clean[:600]


async def gen_hashtags(ctx: JobContext, prose: Dict, images: List[str], max_count: int = 25) -> List[str]:
    """
    Build hashtags in priority layers:
    1. Trill score tags (evidence-based, from telemetry_stage)
    2. Real location tags (city, state, road from pandas/Nominatim)
    3. Bucket personality tags
    4. Platform discovery tags
    5. AI-generated content-specific tags (remaining slots)
    """
    trill: Optional[TrillScore] = ctx.trill or ctx.trill_score
    telem: Optional[TelemetryData] = ctx.telemetry or ctx.telemetry_data

    trill_tags: List[str] = []
    if trill and trill.hashtags:
        trill_tags = [t if t.startswith("#") else f"#{t}" for t in trill.hashtags]

    # Location tags from pandas geocoded data
    location_tags: List[str] = []
    location_context = ""
    if telem:
        city = telem.location_city
        state = telem.location_state
        road = telem.location_road
        display = telem.location_display
        if city:
            location_tags.append(f"#{city.replace(' ', '').replace('-', '')}")
        if state:
            # Try to abbreviate known US states
            state_abbr = _state_abbrev(state)
            if state_abbr.lower() not in (city or "").lower():
                location_tags.append(f"#{state_abbr}")
        if road:
            road_clean = road.replace(" ", "").replace("-", "").replace("/", "")
            if len(road_clean) > 2:
                location_tags.append(f"#{road_clean}")
        if display:
            location_context = f"This clip is from {display}. Include 1-2 real, specific location hashtags for that area."

    # Bucket personality tags
    bucket_tags: Dict[str, List[str]] = {
        "gloryBoy": ["#trillscore", "#gloryboy", "#dashcam", "#carsoftiktok"],
        "euphoric":  ["#trillscore", "#euphoric", "#dashcam"],
        "sendIt":    ["#trillscore", "#sendit", "#dashcam"],
        "spirited":  ["#trillscore", "#spiriteddrive", "#dashcam"],
        "chill":     ["#dashcam", "#driving"],
    }
    bt = bucket_tags.get(prose["trill_bucket"], ["#dashcam"])

    # Platform discovery tags
    platform_tags: List[str] = []
    for p in (ctx.platforms or []):
        pl = p.lower()
        if pl == "tiktok":
            platform_tags += ["#fyp", "#carsoftiktok"]
        elif "youtube" in pl:
            platform_tags += ["#Shorts", "#carsofyoutube"]
        elif pl == "instagram":
            platform_tags += ["#reels", "#carsofinstagram"]
        elif pl == "facebook":
            platform_tags += ["#reels"]

    # How many slots remain for AI after seeded tags
    seeded = list(dict.fromkeys(trill_tags + location_tags + bt + platform_tags))
    ai_slots = max(0, min(max_count - len(seeded), 12))

    ai_tags: List[str] = []
    if ai_slots > 0:
        platform = (ctx.platforms or ["tiktok"])[0].lower()
        system = _build_system(prose["trill_bucket"], platform, "hashtags")
        user = (
            _build_user_prompt(prose, bool(images), ctx)
            + f"\n\n{location_context}"
            + f"\n\nGenerate {ai_slots} hashtags for car/driving content. "
            "Car culture, speed, specific to this clip. "
            "Do NOT include: #blessed #vibes #motivation #goals or generic lifestyle tags. "
            "Return ONLY a JSON array like: [\"#dashcam\", \"#carsoftiktok\"]"
        )
        raw = await _openai(system, user, max_tokens=160, images_b64=images or None, temperature=0.75)
        if raw:
            try:
                clean = raw.strip().strip("`")
                if clean.lower().startswith("json"):
                    clean = clean[4:].strip()
                parsed = json.loads(clean)
                if isinstance(parsed, list):
                    ai_tags = [str(t) for t in parsed if isinstance(t, str) and t.startswith("#")]
            except (json.JSONDecodeError, ValueError):
                ai_tags = [w for w in raw.split() if w.startswith("#")][:ai_slots]

    all_tags = list(dict.fromkeys(trill_tags + location_tags + bt + platform_tags + ai_tags))
    return all_tags[:max_count]


def _state_abbrev(state: str) -> str:
    """Map full US state name to abbreviation for hashtags."""
    abbrevs = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
        "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
        "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
        "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    }
    return abbrevs.get(state.lower(), state[:2].upper())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Generate human-voiced AI title, caption, and hashtags.

    Called after:
    - telemetry_stage   (populates ctx.telemetry + pandas stats)
    - thumbnail_stage   (ctx.thumbnail_path as primary visual)
    - transcode_stage   (ctx.processed_video_path for frame extraction)

    Writes to: ctx.ai_title, ctx.ai_caption, ctx.ai_hashtags
    """
    ctx.mark_stage("caption")

    if not OPENAI_API_KEY:
        raise SkipStage("OPENAI_API_KEY not configured", stage="caption")

    if ctx.entitlements and not getattr(ctx.entitlements, "ai_captions_enabled", True):
        raise SkipStage("AI captions not available for this tier", stage="caption")

    # ── What does the user want generated? ────────────────────────────────────
    has_title = bool(ctx.title and ctx.title.strip())
    has_caption = bool(ctx.caption and ctx.caption.strip())
    has_hashtags = bool(ctx.hashtags)

    auto_captions = ctx.user_settings.get("auto_generate_captions", True)
    auto_captions = ctx.user_settings.get("autoCaptions", auto_captions)
    auto_hashtags = ctx.user_settings.get("auto_generate_hashtags", True)
    auto_hashtags = ctx.user_settings.get("aiHashtagsEnabled", auto_hashtags)
    always_hashtags = ctx.user_settings.get("always_use_hashtags", False)

    do_title = (not has_title) and bool(auto_captions)
    do_caption = (not has_caption) and bool(auto_captions)
    do_hashtags = (not has_hashtags or bool(always_hashtags)) and bool(auto_hashtags)

    if not do_title and not do_caption and not do_hashtags:
        raise SkipStage("All content already present or generation disabled", stage="caption")

    # ── Gather visual frames ──────────────────────────────────────────────────
    video_src: Optional[Path] = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_src = Path(candidate)
            break

    frames: List[Path] = []
    if video_src and ctx.temp_dir:
        try:
            frames = await extract_frames(video_src, ctx.temp_dir, n=8)
        except Exception as e:
            logger.warning(f"[{ctx.upload_id}] Frame extraction failed: {e}")

    # Prepend thumbnail as first frame
    if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
        frames = [Path(ctx.thumbnail_path)] + [f for f in frames if f != ctx.thumbnail_path]

    images_b64 = frames_to_b64(frames, max_frames=6)
    has_frames = len(images_b64) > 0

    # ── Translate telemetry via pandas stats → natural language ───────────────
    prose = telem_to_prose(ctx)
    has_telem = bool(prose["location"] or prose["speed_phrase"] or prose["trill_score"])

    if not has_frames and not has_telem and not ctx.filename and not ctx.title:
        raise SkipStage("No evidence for caption generation", stage="caption")

    logger.info(
        f"[{ctx.upload_id}] Caption engine | bucket={prose['trill_bucket']} | "
        f"location={prose['location']} | speed={prose['speed_phrase']} | "
        f"accel_events={prose['accel_note']} | frames={len(frames)} | "
        f"generate: title={do_title} caption={do_caption} hashtags={do_hashtags}"
    )

    # ── Max hashtags — tier-based limits ──────────────────────────────────────
    max_tags = 15
    if ctx.entitlements:
        tier = getattr(ctx.entitlements, "tier", "free")
        if tier in ("master_admin", "friends_family", "lifetime"):
            max_tags = 50
        elif getattr(ctx.entitlements, "can_ai", False):
            max_tags = 30
    max_tags = min(max_tags, int(ctx.user_settings.get("maxHashtags", max_tags)))

    # ── Fire generators concurrently where safe ────────────────────────────────
    title_task = gen_title(ctx, prose, images_b64) if do_title else None
    caption_task = gen_caption(ctx, prose, images_b64) if do_caption else None
    hashtag_task = gen_hashtags(ctx, prose, images_b64, max_count=max_tags) if do_hashtags else None

    tasks = [t for t in [title_task, caption_task, hashtag_task] if t is not None]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    idx = 0
    if title_task is not None:
        r = results[idx]; idx += 1
        if isinstance(r, str) and r:
            ctx.ai_title = r
            logger.info(f"[{ctx.upload_id}] Title: {ctx.ai_title}")
        elif isinstance(r, Exception):
            logger.warning(f"[{ctx.upload_id}] Title gen error: {r}")

    if caption_task is not None:
        r = results[idx]; idx += 1
        if isinstance(r, str) and r:
            ctx.ai_caption = r
            logger.info(f"[{ctx.upload_id}] Caption: {ctx.ai_caption[:80]}")
        elif isinstance(r, Exception):
            logger.warning(f"[{ctx.upload_id}] Caption gen error: {r}")

    if hashtag_task is not None:
        r = results[idx]; idx += 1
        if isinstance(r, list):
            ctx.ai_hashtags = r
            logger.info(f"[{ctx.upload_id}] Hashtags ({len(r)}): {r[:5]}")
        elif isinstance(r, Exception):
            logger.warning(f"[{ctx.upload_id}] Hashtag gen error: {r}")

    return ctx
