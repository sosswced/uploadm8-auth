"""
Caption creative knobs — single source of truth
==============================================

Style / tone / voice allowlists, UI option metadata, and M8 creative directives
live here. Routers, preference persistence, caption_stage, and m8_engine MUST
import from this module — do not hardcode parallel tuples elsewhere.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

# Caption STYLE = structural architecture of the line.
#
# Every entry carries two layers:
#   - "blueprint": rich prose contract (legacy consumers + deep elaboration)
#   - "facets": machine-composable levers consumed by compose_creative_directive().
#     STYLE owns ONLY these levers: architecture, hook mechanic, length band,
#     timeline-beat plan, and variant rotation. It never dictates emotional heat
#     (TONE's job) or pronouns/diction (VOICE's job).
STYLE_DIRECTIVES: Dict[str, Dict[str, Any]] = {
    "story": {
        "label": "STORY — narrative arc",
        "ui_label": "Story — narrative arc from start to finish",
        "blueprint": (
            "Build a micro-arc from scene_graph.timeline + hydration_story — not a template about 'a video'. "
            "Open on a concrete early beat (HUD time, place, first speed sample, opening OCR), pivot on a mid/peak "
            "beat (trusted MPH sample, landmark, music drop, on-screen text), close on a late beat. "
            "Caption length 150–320 characters. Connective momentum; no bullet fragments. "
            "Each of the 5 variants must enter/exit on DIFFERENT timeline beats so they feel like five retellings."
        ),
        "facets": {
            "architecture": "three-beat micro-arc: entry beat → pivot/peak beat → exit beat, with connective momentum (no bullet fragments)",
            "hook": "open on a concrete EARLY timeline beat (HUD time, place, first speed sample, opening OCR) — never a summary of 'a video'",
            "length": "150–320 characters",
            "beats": "consume at least 3 ordered beats from scene_graph.timeline / hydration_story: early entry, mid/peak pivot, late close",
            "rotation": "each variant must enter AND exit on different timeline beats so the 5 read as five retellings",
        },
    },
    "punchy": {
        "label": "PUNCHY — hook in first 3 words",
        "ui_label": "Punchy — hook in first 3 words, short and viral",
        "blueprint": (
            "Front-load the single most arresting CONCRETE fact in the first 3 words "
            "(a number, a place, a named object, a speed). One or two short lines, telegraphic rhythm, "
            "cut every connective and hedge ('just', 'really', 'kind of'). Under 120 characters. "
            "No narrative ramp — impact then stop. Across the 5 variants, rotate WHICH evidence token leads "
            "(speed → place → object → audio → trill) so no two hooks open on the same word class."
        ),
        "facets": {
            "architecture": "one or two telegraphic lines — impact then stop; cut every connective and hedge",
            "hook": "the single most arresting CONCRETE fact lands inside the first 3 words (number, place, named object, speed)",
            "length": "under 120 characters",
            "beats": "exactly 1–2 beats — only the strongest evidence tokens survive the cut",
            "rotation": "rotate which evidence class leads (speed → place → object → audio → trill) so no two hooks open on the same word class",
        },
    },
    "factual": {
        "label": "FACTUAL — lead with the strongest stat",
        "ui_label": "Factual — lead with the most impressive stat or data point",
        "blueprint": (
            "Lead with the single most impressive VERIFIABLE data point in the evidence — a trusted MPH sample, "
            "a count, a precise place/road, a HUD date/time, an artist/title. Data-forward, zero fluff. "
            "100–220 characters. State the metric, then one tight line of grounded context from the timeline. "
            "Across the 5 variants, lead with a different verified figure each time."
        ),
        "facets": {
            "architecture": "metric-first statement, then ONE tight line of grounded context from the timeline",
            "hook": "the most impressive VERIFIABLE data point leads (trusted MPH sample, count, precise place/road, HUD date/time, artist/title)",
            "length": "100–220 characters",
            "beats": "1 headline metric + 1 supporting timeline beat; zero fluff between them",
            "rotation": "each variant leads with a DIFFERENT verified figure",
        },
    },
    "diary": {
        "label": "DIARY — first-person log of what happened",
        "ui_label": "Diary — first-person log of HUD / timeline beats",
        "blueprint": (
            "Write like a dated field note from inside the clip: HUD clock → place → what changed next. "
            "Use at least two timestamped or ordered beats from scene_graph.timeline (e.g. early MPH sample, "
            "later MPH sample, music ID). 140–280 characters. Intimate, sequential, never omniscient filler."
        ),
        "facets": {
            "architecture": "dated field-note sequence: HUD clock → place → what changed next; intimate and sequential, never omniscient filler",
            "hook": "a timestamp / HUD-clock or ordered opening beat anchors the entry",
            "length": "140–280 characters",
            "beats": "at least 2 timestamped or ordered timeline beats, kept in chronological order",
            "rotation": "vary which timestamps anchor each variant's log entry",
        },
    },
    "listicle": {
        "label": "LISTICLE — stacked evidence beats",
        "ui_label": "Listicle — stacked speed · place · music beats",
        "blueprint": (
            "Stack 3 short grounded beats separated by · or / or line breaks: speed sample · place · music/driver. "
            "Every beat must be a token from hydration_story or timeline. Under 200 characters. No prose throat-clearing "
            "('The video is…'). Rotate which evidence leads across the 5 variants."
        ),
        "facets": {
            "architecture": "3 stacked short beats separated by · or / or line breaks; no prose throat-clearing",
            "hook": "the first stacked beat is the strongest evidence token",
            "length": "under 200 characters",
            "beats": "every list item IS a token from hydration_story or the timeline — no connective prose",
            "rotation": "rotate which evidence class leads the stack across the 5 variants",
        },
    },
    "freestyle": {
        "label": "FREESTYLE — no rails, hydration-first invention of shape",
        "ui_label": "Freestyle — no rails; invent shape from hydration + timeline",
        "blueprint": (
            "NO fixed arc, NO 'first 3 words' rule, NO forced length band. Invent a fresh title/caption SHAPE "
            "for this clip using scene_graph.hydration_story + timeline as the raw material. "
            "You may open mid-scene, end on a question, braid music with speed samples, or lead with driver/HUD time — "
            "as long as every proper noun, MPH figure, place, and song claim is evidenced. "
            "Ban generic wrappers: 'The video is a…', 'high-energy first-person dashcam', 'capturing a journey along'. "
            "The 5 variants must differ in STRUCTURE (not just synonyms)."
        ),
        "facets": {
            "architecture": "invent a fresh SHAPE per variant (question, mid-scene entry, music/speed braid, diary stamp) — no fixed arc",
            "hook": "free — may open mid-scene, on a question, or on driver/HUD time; generic wrappers stay banned",
            "length": "free (soft rails; platform limits still apply)",
            "beats": "hydration_story + timeline are raw material; evidence density mandatory, ordering optional",
            "rotation": "the 5 variants must differ in STRUCTURE itself, not just synonyms",
        },
    },
}

# Caption TONE = emotional register.
#
# TONE facets own ONLY the temperature levers: intensity (1 = flattest,
# 5 = hottest), pacing, punctuation policy, and word-field. TONE never changes
# the structure (STYLE's job) or the speaker's pronouns/diction (VOICE's job).
TONE_DIRECTIVES: Dict[str, Dict[str, Any]] = {
    "authentic": {
        "label": "AUTHENTIC — real talk, first-person, no fluff",
        "ui_label": "Authentic — real talk, first-person, no fluff",
        "register": (
            "Human and direct; first-person or close second-person; plain words over marketing speak. "
            "Sound like a real person who was actually there. One honest observation beats a manufactured hook. "
            "Ban influencer filler ('okay guys', 'here's the thing', 'let me tell you'). No exclamation inflation."
        ),
        "facets": {
            "intensity": 2,
            "pacing": "natural speech rhythm, unforced; one honest observation beats a manufactured hook",
            "punctuation": "no exclamation inflation; plain periods and commas",
            "word_field": "plain words over marketing speak; ban influencer filler ('okay guys', 'here's the thing', 'let me tell you')",
        },
    },
    "hype": {
        "label": "HYPE — high energy, power words, stop-the-scroll",
        "ui_label": "Hype — high energy, power words, stop-the-scroll",
        "register": (
            "High momentum and conviction: strong verbs, tight clauses, forward pull, occasional emphatic word — "
            "still believable. Scale the intensity to the actual subject (a quiet craft gets urgent clarity, not "
            "party-bro shouting). Every spike of energy must trace to something literally on screen or in the audio. "
            "Never invent stakes the footage does not earn."
        ),
        "facets": {
            "intensity": 4,
            "pacing": "high momentum: tight clauses, forward pull; scale intensity to the actual subject",
            "punctuation": "occasional emphatic mark allowed — never stacked (!!), never on invented stakes",
            "word_field": "strong verbs and power nouns; every spike of energy must trace to something literally on screen or in the audio",
        },
    },
    "cinematic": {
        "label": "CINEMATIC — poetic, atmospheric, film-trailer feel",
        "ui_label": "Cinematic — poetic, atmospheric, film trailer feel",
        "register": (
            "Scene-led, sensory language: light, shadow, motion, scale, texture — only what the frames support. "
            "Present tense where it heightens immediacy; trailer-like rhythm without melodrama or clichés that could "
            "apply to any clip. Every image must tether to a visible detail or a spoken line. Restraint over purple prose."
        ),
        "facets": {
            "intensity": 3,
            "pacing": "trailer-like rhythm; present tense where it heightens immediacy",
            "punctuation": "ellipses / dashes sparingly for atmosphere; no melodrama",
            "word_field": "sensory language (light, shadow, motion, scale, texture) tethered to a visible detail or spoken line; restraint over purple prose",
        },
    },
    "calm": {
        "label": "CALM — measured, confident, let the footage speak",
        "ui_label": "Calm — measured, confident, let the footage speak",
        "register": (
            "Measured, breathable pacing; let concrete details carry the weight. Understatement over exclamation; "
            "cool, trustworthy register. No urgency theatrics. Confidence shown through specificity, not volume."
        ),
        "facets": {
            "intensity": 1,
            "pacing": "measured, breathable; let concrete details carry the weight",
            "punctuation": "no exclamation marks; understatement over emphasis",
            "word_field": "specific, cool, trustworthy nouns over adjectives; confidence through specificity, not volume",
        },
    },
    "documentary": {
        "label": "DOCUMENTARY — observational, reportorial",
        "ui_label": "Documentary — observational, reportorial",
        "register": (
            "Observational and precise: report what the HUD, GPS, and audio actually show. Prefer time, place, "
            "speed samples, and named music over vibes. Third-person or neutral first-person. No petrolhead filler."
        ),
        "facets": {
            "intensity": 2,
            "pacing": "even, reportorial cadence; observation before interpretation",
            "punctuation": "neutral; no rhetorical marks or dramatics",
            "word_field": "time / place / measurement vocabulary from HUD, GPS, and audio; zero vibes words, no petrolhead filler",
        },
    },
    "dry": {
        "label": "DRY — deadpan, understated wit",
        "ui_label": "Dry — deadpan, understated wit",
        "register": (
            "Deadpan delivery: let absurd or intense facts (triple-digit MPH, named track, specific town) land "
            "without hype adjectives. Dry humor only when the evidence earns it. Short clauses, cool distance."
        ),
        "facets": {
            "intensity": 2,
            "pacing": "short clauses with a beat of silence between facts; cool distance",
            "punctuation": "flat periods — the joke is the fact, not the mark",
            "word_field": "no hype adjectives; let absurd or intense facts land bare; dry humor only when the evidence earns it",
        },
    },
    "chaotic": {
        "label": "CHAOTIC — kinetic, clipped, interruptive",
        "ui_label": "Chaotic — kinetic, clipped, interruptive",
        "register": (
            "Kinetic and interruptive: fragments OK, mid-thought jumps OK, but every fragment must be an evidence "
            "token (MPH sample, place, song, driver, HUD clock). Energy from pacing, not invented drama."
        ),
        "facets": {
            "intensity": 5,
            "pacing": "interruptive: fragments OK, mid-thought jumps OK — energy comes from pacing, not invented drama",
            "punctuation": "hard cuts, dashes, fragments; never trailing filler",
            "word_field": "every fragment must be an evidence token (MPH sample, place, song, driver, HUD clock)",
        },
    },
}

# Caption VOICE / PERSONA = who is speaking.
#
# VOICE facets own ONLY the speaker levers: point of view (pronouns), diction,
# sentence habits, and one signature move. VOICE never changes structure or
# length (STYLE's job) and never changes emotional heat (TONE's job).
VOICE_DIRECTIVES: Dict[str, Dict[str, Any]] = {
    "default": {
        "label": "DEFAULT — balanced, platform-friendly creator",
        "ui_label": "Default",
        "ui_desc": "Balanced, platform-friendly",
        "persona": (
            "Balanced creator voice: clear hook, specific middle, satisfying close. Confident but not performative. "
            "Match slang and terminology to what the content actually is (chef terms for food, dev terms for code, "
            "driver terms for a drive). Neutral, broadly likeable point of view."
        ),
        "facets": {
            "pov": "neutral, broadly likeable creator; first or close second person as the content suggests",
            "diction": "match terminology to what the content actually is (chef terms for food, driver terms for a drive)",
            "habits": "clear hook, specific middle, satisfying close",
            "signature": "confident but never performative",
        },
    },
    "mentor": {
        "label": "MENTOR — wise, educational, authority",
        "ui_label": "Mentor",
        "ui_desc": "Wise, educational, authority",
        "persona": (
            "Experienced guide: 'you'-oriented, encouraging, zero condescension. Imply expertise through precise "
            "specifics, never a credentials flex. When the clip teaches or demonstrates anything, land one usable "
            "takeaway. Calm authority — the voice of someone who has done this many times."
        ),
        "facets": {
            "pov": "second person 'you'-oriented, experienced guide",
            "diction": "precise specifics and practical verbs; zero condescension, never a credentials flex",
            "habits": "land one usable takeaway when the clip teaches or demonstrates anything",
            "signature": "the calm authority of someone who has done this many times",
        },
    },
    "hypebeast": {
        "label": "HYPEBEAST — all-caps energy, slang, viral",
        "ui_label": "Hypebeast",
        "ui_desc": "All caps energy, slang, viral",
        "persona": (
            "Peak short-form energy: clipped sentences, rhythm, street/viral cadence, sparing ALL-CAPS on the one "
            "word that matters. Slang only when it fits the subject and platform — never empty viral filler "
            "('this is insane', 'no way'). All the hype must trace to a real on-screen or audio moment."
        ),
        "facets": {
            "pov": "street-energy first person, talking to the feed",
            "diction": "street/viral cadence; slang only when it fits the subject; sparing ALL-CAPS on the one word that matters",
            "habits": "clipped sentences with rhythm; never empty viral filler ('this is insane', 'no way')",
            "signature": "every ounce of hype traces to a real on-screen or audio moment",
        },
    },
    "best_friend": {
        "label": "BEST FRIEND — casual, real, relatable",
        "ui_label": "Best Friend",
        "ui_desc": "Casual, real, relatable",
        "persona": (
            "Warm, unfiltered peer texting you about something cool: conversational fragments OK, light self-aware "
            "humor when the content allows, relatable aside. Never mean-spirited or faux-chaos. Reads like a friend, "
            "not a brand. Second-person ('you') and shared-moment framing welcome."
        ),
        "facets": {
            "pov": "second person 'you', warm peer texting a friend; shared-moment framing welcome",
            "diction": "conversational fragments OK; light self-aware humor when the content allows",
            "habits": "one relatable aside; never mean-spirited or faux-chaos",
            "signature": "reads like a friend, never a brand",
        },
    },
    "teacher": {
        "label": "TEACHER — clear, informative, structured",
        "ui_label": "Teacher",
        "ui_desc": "Clear, informative, structured",
        "persona": (
            "Educator clarity: one central idea, a logical mini-arc, minimal jargon unless the visuals clearly expect "
            "it. If the clip is not instructional, still be precise — teach what happened or what to notice in the "
            "footage, not an unrelated life lesson. Structure and signposting over flourish."
        ),
        "facets": {
            "pov": "structured educator addressing learners; neutral person",
            "diction": "minimal jargon unless the visuals expect it; signposting words over flourish",
            "habits": "one central idea carried through a logical mini-arc",
            "signature": "teach what happened or what to notice in the footage — never an unrelated life lesson",
        },
    },
    "cinematic_narrator": {
        "label": "CINEMATIC — film narrator, epic, atmospheric",
        "ui_label": "Cinematic",
        "ui_desc": "Film narrator, epic, atmospheric",
        "persona": (
            "Third-person / omniscient trailer narrator: declarative, image-stacking, slightly elevated register. "
            "Anchored to real events in the clip — no epic narration of nothing happening. Reserve the biggest "
            "flourish for the genuine peak in the footage. Think voiceover, not influencer."
        ),
        "facets": {
            "pov": "third-person omniscient trailer narrator",
            "diction": "declarative, image-stacking, slightly elevated register",
            "habits": "reserve the biggest flourish for the genuine peak in the footage",
            "signature": "voiceover, not influencer — no epic narration of nothing happening",
        },
    },
    "radio_host": {
        "label": "RADIO HOST — drive-time DJ energy",
        "ui_label": "Radio Host",
        "ui_desc": "Drive-time DJ, track + town",
        "persona": (
            "Drive-time host: call out the track, the town, and the speed like a live break — punchy intros, "
            "warm banter cadence, never fake caller bits. Music + place + motion from the Scene Graph only."
        ),
        "facets": {
            "pov": "live drive-time host addressing listeners on-air",
            "diction": "track + town + speed callouts; warm banter cadence",
            "habits": "punchy intro like a live break; never fake caller bits",
            "signature": "music + place + motion, all from the Scene Graph only",
        },
    },
    "journalist": {
        "label": "JOURNALIST — tight lede, who/what/where",
        "ui_label": "Journalist",
        "ui_desc": "Tight lede, who/what/where",
        "persona": (
            "News lede voice: who/what/where/when from HUD + geo + music ID. Neutral verbs, specific nouns, "
            "no hype adjectives. Think wire-copy tightness with one vivid detail from the timeline."
        ),
        "facets": {
            "pov": "neutral third-person wire reporter",
            "diction": "neutral verbs, specific nouns, no hype adjectives",
            "habits": "who/what/where/when lede built from HUD + geo + music ID",
            "signature": "wire-copy tightness with one vivid detail from the timeline",
        },
    },
    "passenger": {
        "label": "PASSENGER — shotgun seat, you-are-there",
        "ui_label": "Passenger",
        "ui_desc": "Shotgun seat, you-are-there",
        "persona": (
            "Shotgun-seat witness: 'we're doing X MPH near Y with Z on the speakers' energy. Present tense, "
            "body-in-the-cabin details only when visible (HUD, road, cabin cues). Never invent passengers or drama."
        ),
        "facets": {
            "pov": "first-person present tense from the shotgun seat — 'we're doing X near Y with Z on the speakers'",
            "diction": "cabin-visible details only (HUD, road, speaker cues)",
            "habits": "you-are-there immediacy; ride the moment as it happens",
            "signature": "never invent passengers or drama",
        },
    },
}

# Derived allowlists — never re-list these keys elsewhere.
CAPTION_STYLES: Tuple[str, ...] = tuple(STYLE_DIRECTIVES.keys())
CAPTION_TONES: Tuple[str, ...] = tuple(TONE_DIRECTIVES.keys())
CAPTION_VOICES: Tuple[str, ...] = tuple(VOICE_DIRECTIVES.keys())

DEFAULT_CAPTION_STYLE = "story"
DEFAULT_CAPTION_TONE = "authentic"
DEFAULT_CAPTION_VOICE = "default"

# Strategy-slug → UI voice (policy collapse → rich directive).
PERSONA_SLUG_TO_VOICE_UI: Dict[str, str] = {
    "storyteller": "cinematic_narrator",
    "creator_coach": "mentor",
    "hype_friend": "hypebeast",
    "expert_analyst": "teacher",
    "radio_host": "radio_host",
    "journalist": "journalist",
    "passenger": "passenger",
}


def normalize_caption_style(value: Any, *, default: str = DEFAULT_CAPTION_STYLE) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    return v if v in STYLE_DIRECTIVES else default


def normalize_caption_tone(value: Any, *, default: str = DEFAULT_CAPTION_TONE) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    return v if v in TONE_DIRECTIVES else default


def normalize_caption_voice(value: Any, *, default: str = DEFAULT_CAPTION_VOICE) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    if v in VOICE_DIRECTIVES:
        return v
    mapped = PERSONA_SLUG_TO_VOICE_UI.get(v)
    if mapped and mapped in VOICE_DIRECTIVES:
        return mapped
    return default


def style_directive(style_ui: str) -> Dict[str, Any]:
    return STYLE_DIRECTIVES[normalize_caption_style(style_ui)]


def tone_directive(tone_ui: str) -> Dict[str, Any]:
    return TONE_DIRECTIVES[normalize_caption_tone(tone_ui)]


def voice_directive(voice_ui: str) -> Dict[str, Any]:
    return VOICE_DIRECTIVES[normalize_caption_voice(voice_ui)]


# ─────────────────────────────────────────────────────────────────────────────
# Combinatorial composer
#
# Every (style, tone, voice) triple composes into ONE coherent brief instead of
# three stacked paragraphs. Ownership is strict:
#   STYLE  → architecture, hook mechanic, length band, beat plan, rotation
#   TONE   → intensity (1–5), pacing, punctuation, word-field
#   VOICE  → point of view, diction, sentence habits, signature move
# Interaction rules below are DERIVED from the facets, so adding a new style,
# tone, or voice to the registries automatically yields new valid combinations
# (currently len(styles) × len(tones) × len(voices)) with no extra wiring.
# ─────────────────────────────────────────────────────────────────────────────

_COMPACT_STYLES = frozenset({"punchy", "listicle", "factual"})
_ARC_STYLES = frozenset({"story", "diary"})
_FIRST_PERSON_STYLES = frozenset({"diary"})
_DISCIPLINED_VOICES = frozenset({"teacher", "journalist", "mentor"})
_THIRD_PERSON_VOICES = frozenset({"cinematic_narrator", "journalist"})
_EVIDENCE_LEAD_ROTATION: Tuple[str, ...] = (
    "speed/telemetry",
    "place/geo",
    "object/visual",
    "music/audio",
    "trill/energy",
)


def total_combinations() -> int:
    return len(STYLE_DIRECTIVES) * len(TONE_DIRECTIVES) * len(VOICE_DIRECTIVES)


def combination_index(style_ui: str, tone_ui: str, voice_ui: str) -> int:
    """Stable 1-based index of a combo in registry order (for logs/telemetry)."""
    s = normalize_caption_style(style_ui)
    t = normalize_caption_tone(tone_ui)
    v = normalize_caption_voice(voice_ui)
    si = CAPTION_STYLES.index(s)
    ti = CAPTION_TONES.index(t)
    vi = CAPTION_VOICES.index(v)
    return si * len(CAPTION_TONES) * len(CAPTION_VOICES) + ti * len(CAPTION_VOICES) + vi + 1


CAPTION_CREATIVE_PICK_MODES: Tuple[str, ...] = ("off", "random", "cycle")
DEFAULT_CAPTION_CREATIVE_PICK_MODE = "off"


def normalize_caption_creative_pick_mode(value: Any) -> str:
    """Normalize pick mode: off | random | cycle."""
    if value is True or value == 1:
        return "random"
    s = str(value or "").strip().lower().replace("-", "_")
    if s in ("1", "true", "yes", "on", "randomize", "random"):
        return "random"
    if s in ("cycle", "sweep", "combinatorial", "all", "sequential", "shuffle", "deck"):
        return "cycle"
    if s in ("0", "false", "no", "off", "none", ""):
        return "off"
    return DEFAULT_CAPTION_CREATIVE_PICK_MODE


def _pref_bool(us: Dict[str, Any], camel: str, snake: str, *, default: bool) -> bool:
    if camel in us:
        return bool(us.get(camel))
    if snake in us:
        return bool(us.get(snake))
    return default


def parse_caption_creative_vary_flags(prefs: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Per-axis vary flags. Missing keys default to True (vary all when randomize is on).

    Prefer positive ``vary*`` keys; ``lock*`` / ``lockCaption*`` invert when present.
    """
    us = prefs if isinstance(prefs, dict) else {}

    def _axis(vary_camel: str, vary_snake: str, lock_camel: str, lock_snake: str) -> bool:
        if vary_camel in us or vary_snake in us:
            return _pref_bool(us, vary_camel, vary_snake, default=True)
        if lock_camel in us or lock_snake in us:
            return not _pref_bool(us, lock_camel, lock_snake, default=False)
        return True

    return {
        "style": _axis(
            "captionCreativeVaryStyle",
            "caption_creative_vary_style",
            "captionCreativeLockStyle",
            "caption_creative_lock_style",
        ),
        "tone": _axis(
            "captionCreativeVaryTone",
            "caption_creative_vary_tone",
            "captionCreativeLockTone",
            "caption_creative_lock_tone",
        ),
        "voice": _axis(
            "captionCreativeVaryVoice",
            "caption_creative_vary_voice",
            "captionCreativeLockVoice",
            "caption_creative_lock_voice",
        ),
    }


def combination_at(index: int) -> Tuple[str, str, str]:
    """Return (style, tone, voice) for a 0-based index into the full product space."""
    return combination_at_axes(
        index,
        styles=CAPTION_STYLES,
        tones=CAPTION_TONES,
        voices=CAPTION_VOICES,
    )


def combination_at_axes(
    index: int,
    *,
    styles: Tuple[str, ...],
    tones: Tuple[str, ...],
    voices: Tuple[str, ...],
) -> Tuple[str, str, str]:
    """0-based pick inside a (possibly locked) style×tone×voice product."""
    n_s = max(1, len(styles))
    n_t = max(1, len(tones))
    n_v = max(1, len(voices))
    n = n_s * n_t * n_v
    i = int(index) % n
    si = i // (n_t * n_v)
    rem = i % (n_t * n_v)
    ti = rem // n_v
    vi = rem % n_v
    return (styles[si % n_s], tones[ti % n_t], voices[vi % n_v])


def pick_random_combination(
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str]:
    """Pick one uniform random style×tone×voice triple from the registry."""
    return pick_random_combination_axes(
        rng=rng,
        styles=CAPTION_STYLES,
        tones=CAPTION_TONES,
        voices=CAPTION_VOICES,
    )


def pick_random_combination_axes(
    *,
    styles: Tuple[str, ...],
    tones: Tuple[str, ...],
    voices: Tuple[str, ...],
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str]:
    r = rng if rng is not None else random
    n = max(1, len(styles) * len(tones) * len(voices))
    return combination_at_axes(int(r.randrange(n)), styles=styles, tones=tones, voices=voices)


def _stable_combo_index_from_upload_id(upload_id: Any, *, modulus: int) -> int:
    raw = str(upload_id or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:12], 16) % max(1, int(modulus))


def subspace_axes_from_prefs(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the unlocked style/tone/voice product for the current lock state."""
    us = prefs if isinstance(prefs, dict) else {}
    base_style = normalize_caption_style(us.get("captionStyle") or us.get("caption_style"))
    base_tone = normalize_caption_tone(us.get("captionTone") or us.get("caption_tone"))
    base_voice = normalize_caption_voice(us.get("captionVoice") or us.get("caption_voice"))
    vary = parse_caption_creative_vary_flags(us)
    styles: Tuple[str, ...] = CAPTION_STYLES if vary["style"] else (base_style,)
    tones: Tuple[str, ...] = CAPTION_TONES if vary["tone"] else (base_tone,)
    voices: Tuple[str, ...] = CAPTION_VOICES if vary["voice"] else (base_voice,)
    size = max(1, len(styles) * len(tones) * len(voices))
    fingerprint = "|".join(
        [
            "1" if vary["style"] else f"S:{base_style}",
            "1" if vary["tone"] else f"T:{base_tone}",
            "1" if vary["voice"] else f"V:{base_voice}",
            f"n={size}",
        ]
    )
    return {
        "styles": styles,
        "tones": tones,
        "voices": voices,
        "vary": vary,
        "size": size,
        "fingerprint": fingerprint,
        "base_style": base_style,
        "base_tone": base_tone,
        "base_voice": base_voice,
    }


def shuffled_deck_order(seed: str, n: int) -> List[int]:
    """Deterministic Fisher–Yates shuffle of 0..n-1 from a string seed."""
    order = list(range(max(1, int(n))))
    random.Random(str(seed)).shuffle(order)
    return order


def allocate_shuffled_cycle_draw(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """Draw the next combo from a shuffled deck (no file-order, no repeats until reset).

    Mutates ``prefs`` with:
      - captionCreativeComboIndex / caption_creative_combo_index (0-based subspace idx)
      - captionCreativeShuffleSeed / deck cursor / fingerprint (advanced)

    Returns the deck fields that should be persisted on the account for the next draw.
    """
    space = subspace_axes_from_prefs(prefs)
    size = int(space["size"])
    fingerprint = str(space["fingerprint"])

    prev_fp = str(
        prefs.get("captionCreativeDeckFingerprint")
        or prefs.get("caption_creative_deck_fingerprint")
        or ""
    )
    try:
        cursor = int(
            prefs.get("captionCreativeDeckCursor")
            if "captionCreativeDeckCursor" in prefs
            else prefs.get("caption_creative_deck_cursor")
            or 0
        )
    except (TypeError, ValueError):
        cursor = 0

    seed = str(
        prefs.get("captionCreativeShuffleSeed")
        or prefs.get("caption_creative_shuffle_seed")
        or ""
    ).strip()

    # New deck when locks change, seed missing, or deck exhausted.
    if (not seed) or prev_fp != fingerprint or cursor < 0 or cursor >= size:
        seed = hashlib.sha256(
            f"{fingerprint}:{cursor}:{random.randrange(1 << 30)}".encode("utf-8")
        ).hexdigest()[:16]
        cursor = 0

    order = shuffled_deck_order(seed, size)
    raw_idx = int(order[cursor % size])
    next_cursor = cursor + 1
    # When the deck empties, next draw will reshuffle (new seed).
    if next_cursor >= size:
        next_seed = hashlib.sha256(f"{seed}:reshuffle:{fingerprint}".encode("utf-8")).hexdigest()[:16]
        persist_seed = next_seed
        persist_cursor = 0
    else:
        persist_seed = seed
        persist_cursor = next_cursor

    prefs["caption_creative_combo_index"] = prefs["captionCreativeComboIndex"] = raw_idx
    prefs["caption_creative_shuffle_seed"] = prefs["captionCreativeShuffleSeed"] = persist_seed
    prefs["caption_creative_deck_cursor"] = prefs["captionCreativeDeckCursor"] = persist_cursor
    prefs["caption_creative_deck_fingerprint"] = prefs["captionCreativeDeckFingerprint"] = fingerprint
    # Keep the seed that produced THIS draw for artifacts/debug (pre-advance).
    prefs["caption_creative_draw_seed"] = prefs["captionCreativeDrawSeed"] = seed
    prefs["caption_creative_draw_cursor"] = prefs["captionCreativeDrawCursor"] = cursor

    return {
        "captionCreativeShuffleSeed": persist_seed,
        "caption_creative_shuffle_seed": persist_seed,
        "captionCreativeDeckCursor": persist_cursor,
        "caption_creative_deck_cursor": persist_cursor,
        "captionCreativeDeckFingerprint": fingerprint,
        "caption_creative_deck_fingerprint": fingerprint,
        "drawn_subspace_index": raw_idx,
        "draw_seed": seed,
        "draw_cursor": cursor,
        "subspace_size": size,
    }


def resolve_caption_creative_knobs(
    prefs: Optional[Dict[str, Any]],
    *,
    upload_id: Any = None,
    combo_index: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Resolve style/tone/voice with optional per-axis vary/lock.

    Pref keys (snake or camel):
      - randomizeCaptionCreative / captionCreativePickMode (off|random|cycle)
      - captionCreativeComboIndex (0-based subspace index; cycle/shuffle deck)
      - captionCreativeVaryStyle|Tone|Voice (bool; default True when randomizing)
      - captionCreativeLockStyle|Tone|Voice (bool; invert of vary when set)
      - captionStyle / captionTone / captionVoice (locked-axis values)

    Cycle mode expects a server-allocated shuffled deck index (presign), not file order.
    Returns: style, tone, voice, pick_mode, combo_index (1-based in full matrix),
    randomized, vary (dict), subspace_size.
    """
    us = prefs if isinstance(prefs, dict) else {}
    mode_raw = (
        us.get("captionCreativePickMode")
        if "captionCreativePickMode" in us
        else us.get("caption_creative_pick_mode")
    )
    if mode_raw is None:
        mode_raw = (
            us.get("randomizeCaptionCreative")
            if "randomizeCaptionCreative" in us
            else us.get("randomize_caption_creative")
        )
    mode = normalize_caption_creative_pick_mode(mode_raw)

    space = subspace_axes_from_prefs(us)
    base_style = space["base_style"]
    base_tone = space["base_tone"]
    base_voice = space["base_voice"]
    vary = space["vary"]

    if mode == "off" or not any(vary.values()):
        return {
            "style": base_style,
            "tone": base_tone,
            "voice": base_voice,
            "pick_mode": "off" if mode == "off" else mode,
            "combo_index": combination_index(base_style, base_tone, base_voice),
            "randomized": False,
            "vary": {"style": False, "tone": False, "voice": False},
            "subspace_size": 1,
        }

    styles: Tuple[str, ...] = space["styles"]
    tones: Tuple[str, ...] = space["tones"]
    voices: Tuple[str, ...] = space["voices"]
    subspace = int(space["size"])

    if mode == "cycle":
        idx0: Optional[int] = combo_index
        if idx0 is None:
            raw_idx = (
                us.get("captionCreativeComboIndex")
                if "captionCreativeComboIndex" in us
                else us.get("caption_creative_combo_index")
            )
            if raw_idx is not None and str(raw_idx).strip() != "":
                try:
                    idx0 = int(raw_idx)
                except (TypeError, ValueError):
                    idx0 = None
        if idx0 is None:
            # Fallback if presign forgot to allocate: stable hash (not file order).
            idx0 = _stable_combo_index_from_upload_id(upload_id, modulus=subspace)
        style, tone, voice = combination_at_axes(idx0, styles=styles, tones=tones, voices=voices)
    else:
        style, tone, voice = pick_random_combination_axes(
            styles=styles, tones=tones, voices=voices, rng=rng
        )

    return {
        "style": style,
        "tone": tone,
        "voice": voice,
        "pick_mode": mode,
        "combo_index": combination_index(style, tone, voice),
        "randomized": True,
        "vary": dict(vary),
        "subspace_size": subspace,
    }


def _interaction_rules(style_key: str, tone_key: str, voice_key: str) -> List[str]:
    """Deterministic tension-resolution rules derived from the facets.

    These are what make each combination feel COMPOSED rather than three
    directives pasted together: the hot/cold and compact/arc collisions get an
    explicit resolution instead of letting the model pick a winner silently.
    """
    intensity = int(TONE_DIRECTIVES[tone_key]["facets"]["intensity"])
    rules: List[str] = [
        "If two rules collide: structure/length → STYLE wins; emotional heat → TONE wins; pronouns/diction → VOICE wins.",
    ]
    if intensity >= 4 and style_key in _ARC_STYLES:
        rules.append(
            f"High heat ({intensity}/5) inside an arc style: compress — shorter sentences INSIDE the "
            "style's length band and beat plan, never a longer caption."
        )
    if intensity >= 4 and style_key in _COMPACT_STYLES:
        rules.append(
            f"High heat ({intensity}/5) on a compact style: the energy lives in verb choice and cut rhythm; "
            "the character budget does not grow."
        )
    if intensity <= 2 and style_key in _COMPACT_STYLES:
        rules.append(
            f"Low heat ({intensity}/5) on a compact style: keep the compression, strip the urgency — "
            "the hook lands through specificity, not volume."
        )
    if intensity <= 2 and style_key == "freestyle":
        rules.append(
            f"Low heat ({intensity}/5) freestyle: invention shows in the SHAPE, not in energy words."
        )
    if intensity >= 4 and voice_key in _DISCIPLINED_VOICES:
        rules.append(
            f"High heat ({intensity}/5) through a disciplined voice: the speaker stays composed — energy "
            "shows in verbs and pacing, never slang or caps beyond the voice's own rules."
        )
    if intensity <= 2 and voice_key == "hypebeast":
        rules.append(
            f"Hypebeast at low heat ({intensity}/5): keep the cadence and diction, drop the caps and emphatics."
        )
    if style_key in _FIRST_PERSON_STYLES and voice_key in _THIRD_PERSON_VOICES:
        rules.append(
            "POV clash: keep the STYLE's log/diary structure but write it in the VOICE's third-person "
            "pronouns — a field report ABOUT the drive, same beat order."
        )
    rules.append(
        "Swap-test: if any ONE of the three knobs changed, the copy must read audibly different — "
        "different opening rhythm, sentence lengths, and word choices, not the same caption reskinned."
    )
    return rules


def interaction_contract(style_ui: str, tone_ui: str, voice_ui: str) -> str:
    """Bullet list of the composed tension-resolution rules for a combo.

    For consumers (legacy caption_stage prompt) that already print the three
    directives separately and only need the per-combination interaction rules.
    """
    s_key = normalize_caption_style(style_ui)
    t_key = normalize_caption_tone(tone_ui)
    v_key = normalize_caption_voice(voice_ui)
    intensity = int(TONE_DIRECTIVES[t_key]["facets"]["intensity"])
    head = (
        f"- Build the {s_key.upper()} structure, delivered at {t_key.upper()} intensity "
        f"({intensity}/5), spoken as {v_key.upper()}."
    )
    rules = "\n".join(f"- {r}" for r in _interaction_rules(s_key, t_key, v_key))
    return f"{head}\n{rules}"


def compose_creative_directive(
    style_ui: str,
    tone_ui: str,
    voice_ui: str,
    *,
    variant_seed: Optional[int] = None,
) -> str:
    """Compose one coherent creative brief for a (style, tone, voice) triple.

    Output changes materially when ANY single knob changes, and an optional
    ``variant_seed`` (e.g. derived from the upload id) rotates which evidence
    class leads variant 1 so identical settings still produce fresh openings
    run-to-run.
    """
    s_key = normalize_caption_style(style_ui)
    t_key = normalize_caption_tone(tone_ui)
    v_key = normalize_caption_voice(voice_ui)
    style = STYLE_DIRECTIVES[s_key]
    tone = TONE_DIRECTIVES[t_key]
    voice = VOICE_DIRECTIVES[v_key]
    sf, tf, vf = style["facets"], tone["facets"], voice["facets"]
    combo = f"{s_key.upper()} × {t_key.upper()} × {v_key.upper()}"
    idx = combination_index(s_key, t_key, v_key)
    intensity = int(tf["intensity"])

    interaction = "\n".join(f"- {r}" for r in _interaction_rules(s_key, t_key, v_key))

    rotation_line = ""
    if variant_seed is not None:
        lead = _EVIDENCE_LEAD_ROTATION[int(variant_seed) % len(_EVIDENCE_LEAD_ROTATION)]
        rotation_line = (
            f"\nFRESHNESS ROTATION (seed {int(variant_seed)}): variant 1 foregrounds a {lead} token "
            "(when that evidence exists), then continue the style's own rotation order. "
            "This keeps repeat uploads with identical settings from opening the same way.\n"
        )

    return f"""━━ CREATIVE COMBINATION BRIEF — {combo} (combination {idx}/{total_combinations()}) ━━
This is ONE composed contract, not three stacked essays. Each axis owns different levers; apply all
three simultaneously. Delivery only — every fact, name, number still comes from Scene Graph evidence.

ARCHITECTURE — owned by STYLE = {style['label']}:
  structure: {sf['architecture']}
  hook mechanic: {sf['hook']}
  length: {sf['length']}
  beat plan: {sf['beats']}
  variant rotation: {sf['rotation']}
  full contract: {style['blueprint']}

ENERGY — owned by TONE = {tone['label']} (intensity {intensity}/5):
  pacing: {tf['pacing']}
  punctuation: {tf['punctuation']}
  word-field: {tf['word_field']}
  full register: {tone['register']}

SPEAKER — owned by VOICE = {voice['label']}:
  point of view: {vf['pov']}
  diction: {vf['diction']}
  sentence habits: {vf['habits']}
  signature: {vf['signature']}
  full persona: {voice['persona']}

INTERACTION CONTRACT (composed for THIS combination — non-negotiable):
- Build the {s_key.upper()} structure, delivered at {t_key.upper()} intensity ({intensity}/5), spoken as {v_key.upper()}.
{interaction}
- Do NOT fall back to a neutral house voice. The selected voice's diction and point of view must be
  audible in every variant; the tone's temperature must be felt in every sentence.
- Stay evidence-grounded: this brief is HOW it is said; the Scene Graph is WHAT is said.
{rotation_line}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def cell_micro_brief(style_ui: str, tone_ui: str, voice_ui: str) -> str:
    """Compact one-line composed brief for evidence-matrix cells."""
    s_key = normalize_caption_style(style_ui)
    t_key = normalize_caption_tone(tone_ui)
    v_key = normalize_caption_voice(voice_ui)
    sf = STYLE_DIRECTIVES[s_key]["facets"]
    tf = TONE_DIRECTIVES[t_key]["facets"]
    vf = VOICE_DIRECTIVES[v_key]["facets"]
    hook = str(sf["hook"]).split("—")[0].split("(")[0].strip().rstrip(";,. ")
    pacing = str(tf["pacing"]).split(";")[0].split("—")[0].strip().rstrip(";,. ")
    pov = str(vf["pov"]).split("—")[0].split(";")[0].strip().rstrip(";,. ")
    return (
        f"{s_key} structure ({sf['length']}), heat {int(tf['intensity'])}/5 ({pacing}), "
        f"spoken as {pov}; hook: {hook}"
    )


def ui_style_options() -> List[Dict[str, str]]:
    return [
        {"value": k, "label": str(v.get("ui_label") or v.get("label") or k)}
        for k, v in STYLE_DIRECTIVES.items()
    ]


def ui_tone_options() -> List[Dict[str, str]]:
    return [
        {"value": k, "label": str(v.get("ui_label") or v.get("label") or k)}
        for k, v in TONE_DIRECTIVES.items()
    ]


def ui_voice_options() -> List[Dict[str, str]]:
    return [
        {
            "value": k,
            "label": str(v.get("ui_label") or k),
            "desc": str(v.get("ui_desc") or ""),
        }
        for k, v in VOICE_DIRECTIVES.items()
    ]


def evidence_matrix_cell_specs(
    style_ui: str,
    tone_ui: str,
    voice_ui: str,
) -> List[Tuple[str, str, str]]:
    """Sweep every registered style/tone/voice without hardcoding subsets.

    Builds a compact matrix:
      - every style × user tone × user voice
      - user style × every tone × user voice
      - user style × user tone × every voice
    """
    style_ui = normalize_caption_style(style_ui)
    tone_ui = normalize_caption_tone(tone_ui)
    voice_ui = normalize_caption_voice(voice_ui)
    seen: set[Tuple[str, str, str]] = set()
    out: List[Tuple[str, str, str]] = []

    def _add(s: str, t: str, v: str) -> None:
        key = (s, t, v)
        if key not in seen:
            seen.add(key)
            out.append(key)

    for s in CAPTION_STYLES:
        _add(s, tone_ui, voice_ui)
    for t in CAPTION_TONES:
        _add(style_ui, t, voice_ui)
    for v in CAPTION_VOICES:
        _add(style_ui, tone_ui, v)
    return out


def trusted_peak_speed_mph(
    *,
    telemetry_max: float = 0.0,
    osd_max: float = 0.0,
    series_peak: float = 0.0,
    vision_peak: float = 0.0,
    spike_delta_mph: float = 35.0,
) -> Tuple[float, str]:
    """Resolve publishable peak MPH.

    Priority: .map telemetry (never capped by OSD samples) → OSD aggregate
    (capped by trusted series when it looks like an OCR spike) → series → vision.
    """
    try:
        tel = float(telemetry_max or 0)
    except (TypeError, ValueError):
        tel = 0.0
    try:
        osd = float(osd_max or 0)
    except (TypeError, ValueError):
        osd = 0.0
    try:
        series = float(series_peak or 0)
    except (TypeError, ValueError):
        series = 0.0
    try:
        vision = float(vision_peak or 0)
    except (TypeError, ValueError):
        vision = 0.0

    if tel >= 5:
        return tel, "telemetry"
    if osd >= 5:
        if series >= 5 and osd > series + spike_delta_mph:
            return series, "osd+series_cap"
        return osd, "osd"
    if series >= 5:
        return series, "osd_series"
    if vision >= 5:
        if series >= 5 and vision > series + spike_delta_mph:
            return series, "vision_ocr+series_cap"
        return vision, "vision_ocr"
    return 0.0, ""


def osd_series_peak_mph(osd: Optional[Dict[str, Any]]) -> float:
    """Max trusted HUD sample from dashcam_osd_context.speed_series / samples."""
    if not isinstance(osd, dict) or not osd or osd.get("skipped"):
        return 0.0
    peak = 0.0
    series = osd.get("speed_series") if isinstance(osd.get("speed_series"), list) else []
    for entry in series:
        if not isinstance(entry, dict):
            continue
        try:
            peak = max(peak, float(entry.get("mph") or entry.get("speed_mph") or 0))
        except (TypeError, ValueError):
            continue
    if peak >= 5:
        return peak
    for s in (osd.get("samples") or []):
        if not isinstance(s, dict) or not s.get("speed_hud_anchored"):
            continue
        try:
            peak = max(peak, float(s.get("speed_mph") or 0))
        except (TypeError, ValueError):
            continue
    return peak


__all__ = [
    "STYLE_DIRECTIVES",
    "TONE_DIRECTIVES",
    "VOICE_DIRECTIVES",
    "CAPTION_STYLES",
    "CAPTION_TONES",
    "CAPTION_VOICES",
    "DEFAULT_CAPTION_STYLE",
    "DEFAULT_CAPTION_TONE",
    "DEFAULT_CAPTION_VOICE",
    "PERSONA_SLUG_TO_VOICE_UI",
    "normalize_caption_style",
    "normalize_caption_tone",
    "normalize_caption_voice",
    "style_directive",
    "tone_directive",
    "voice_directive",
    "total_combinations",
    "combination_index",
    "combination_at",
    "combination_at_axes",
    "pick_random_combination",
    "pick_random_combination_axes",
    "parse_caption_creative_vary_flags",
    "subspace_axes_from_prefs",
    "shuffled_deck_order",
    "allocate_shuffled_cycle_draw",
    "normalize_caption_creative_pick_mode",
    "resolve_caption_creative_knobs",
    "CAPTION_CREATIVE_PICK_MODES",
    "DEFAULT_CAPTION_CREATIVE_PICK_MODE",
    "compose_creative_directive",
    "interaction_contract",
    "cell_micro_brief",
    "ui_style_options",
    "ui_tone_options",
    "ui_voice_options",
    "evidence_matrix_cell_specs",
    "trusted_peak_speed_mph",
    "osd_series_peak_mph",
]
