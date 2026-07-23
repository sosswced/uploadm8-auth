"""
Caption creative knobs — single source of truth
==============================================

Style / tone / voice allowlists, UI option metadata, and M8 creative directives
live here. Routers, preference persistence, caption_stage, and m8_engine MUST
import from this module — do not hardcode parallel tuples elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Caption STYLE = structural architecture of the line.
STYLE_DIRECTIVES: Dict[str, Dict[str, str]] = {
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
    },
    "diary": {
        "label": "DIARY — first-person log of what happened",
        "ui_label": "Diary — first-person log of HUD / timeline beats",
        "blueprint": (
            "Write like a dated field note from inside the clip: HUD clock → place → what changed next. "
            "Use at least two timestamped or ordered beats from scene_graph.timeline (e.g. early MPH sample, "
            "later MPH sample, music ID). 140–280 characters. Intimate, sequential, never omniscient filler."
        ),
    },
    "listicle": {
        "label": "LISTICLE — stacked evidence beats",
        "ui_label": "Listicle — stacked speed · place · music beats",
        "blueprint": (
            "Stack 3 short grounded beats separated by · or / or line breaks: speed sample · place · music/driver. "
            "Every beat must be a token from hydration_story or timeline. Under 200 characters. No prose throat-clearing "
            "('The video is…'). Rotate which evidence leads across the 5 variants."
        ),
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
    },
}

# Caption TONE = emotional register.
TONE_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "authentic": {
        "label": "AUTHENTIC — real talk, first-person, no fluff",
        "ui_label": "Authentic — real talk, first-person, no fluff",
        "register": (
            "Human and direct; first-person or close second-person; plain words over marketing speak. "
            "Sound like a real person who was actually there. One honest observation beats a manufactured hook. "
            "Ban influencer filler ('okay guys', 'here's the thing', 'let me tell you'). No exclamation inflation."
        ),
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
    },
    "cinematic": {
        "label": "CINEMATIC — poetic, atmospheric, film-trailer feel",
        "ui_label": "Cinematic — poetic, atmospheric, film trailer feel",
        "register": (
            "Scene-led, sensory language: light, shadow, motion, scale, texture — only what the frames support. "
            "Present tense where it heightens immediacy; trailer-like rhythm without melodrama or clichés that could "
            "apply to any clip. Every image must tether to a visible detail or a spoken line. Restraint over purple prose."
        ),
    },
    "calm": {
        "label": "CALM — measured, confident, let the footage speak",
        "ui_label": "Calm — measured, confident, let the footage speak",
        "register": (
            "Measured, breathable pacing; let concrete details carry the weight. Understatement over exclamation; "
            "cool, trustworthy register. No urgency theatrics. Confidence shown through specificity, not volume."
        ),
    },
    "documentary": {
        "label": "DOCUMENTARY — observational, reportorial",
        "ui_label": "Documentary — observational, reportorial",
        "register": (
            "Observational and precise: report what the HUD, GPS, and audio actually show. Prefer time, place, "
            "speed samples, and named music over vibes. Third-person or neutral first-person. No petrolhead filler."
        ),
    },
    "dry": {
        "label": "DRY — deadpan, understated wit",
        "ui_label": "Dry — deadpan, understated wit",
        "register": (
            "Deadpan delivery: let absurd or intense facts (triple-digit MPH, named track, specific town) land "
            "without hype adjectives. Dry humor only when the evidence earns it. Short clauses, cool distance."
        ),
    },
    "chaotic": {
        "label": "CHAOTIC — kinetic, clipped, interruptive",
        "ui_label": "Chaotic — kinetic, clipped, interruptive",
        "register": (
            "Kinetic and interruptive: fragments OK, mid-thought jumps OK, but every fragment must be an evidence "
            "token (MPH sample, place, song, driver, HUD clock). Energy from pacing, not invented drama."
        ),
    },
}

# Caption VOICE / PERSONA = who is speaking.
VOICE_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "default": {
        "label": "DEFAULT — balanced, platform-friendly creator",
        "ui_label": "Default",
        "ui_desc": "Balanced, platform-friendly",
        "persona": (
            "Balanced creator voice: clear hook, specific middle, satisfying close. Confident but not performative. "
            "Match slang and terminology to what the content actually is (chef terms for food, dev terms for code, "
            "driver terms for a drive). Neutral, broadly likeable point of view."
        ),
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
    },
    "radio_host": {
        "label": "RADIO HOST — drive-time DJ energy",
        "ui_label": "Radio Host",
        "ui_desc": "Drive-time DJ, track + town",
        "persona": (
            "Drive-time host: call out the track, the town, and the speed like a live break — punchy intros, "
            "warm banter cadence, never fake caller bits. Music + place + motion from the Scene Graph only."
        ),
    },
    "journalist": {
        "label": "JOURNALIST — tight lede, who/what/where",
        "ui_label": "Journalist",
        "ui_desc": "Tight lede, who/what/where",
        "persona": (
            "News lede voice: who/what/where/when from HUD + geo + music ID. Neutral verbs, specific nouns, "
            "no hype adjectives. Think wire-copy tightness with one vivid detail from the timeline."
        ),
    },
    "passenger": {
        "label": "PASSENGER — shotgun seat, you-are-there",
        "ui_label": "Passenger",
        "ui_desc": "Shotgun seat, you-are-there",
        "persona": (
            "Shotgun-seat witness: 'we're doing X MPH near Y with Z on the speakers' energy. Present tense, "
            "body-in-the-cabin details only when visible (HUD, road, cabin cues). Never invent passengers or drama."
        ),
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


def style_directive(style_ui: str) -> Dict[str, str]:
    return STYLE_DIRECTIVES[normalize_caption_style(style_ui)]


def tone_directive(tone_ui: str) -> Dict[str, str]:
    return TONE_DIRECTIVES[normalize_caption_tone(tone_ui)]


def voice_directive(voice_ui: str) -> Dict[str, str]:
    return VOICE_DIRECTIVES[normalize_caption_voice(voice_ui)]


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
    "ui_style_options",
    "ui_tone_options",
    "ui_voice_options",
    "evidence_matrix_cell_specs",
    "trusted_peak_speed_mph",
    "osd_series_peak_mph",
]
