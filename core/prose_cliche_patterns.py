"""
Prose cliché patterns — single source of truth
=============================================

Stock AI / travel / motorsport openers used by:

* ``services.hydration_enforcer`` — wipe generic captions/titles
* ``stages.m8_engine`` — ranking penalty + quality-gate harden
* ``core.caption_creative`` / M8 prompts — anti-generic ban text
* ``tools.backtest_prose_hype_cliches`` — live opener scans

Orthogonal systems (do NOT fold into this module):

* ``is_formula_stub_caption`` — MPH receipts / Anchored / Captured / Through–with
* ``generic_hard_ban`` — Vision taxonomy slugs / colors

Diagnosis freeze (stub vs generic vs wipe labels):

* stub / receipt → ``is_formula_stub_caption`` (MPH checklist / timeline receipts)
* generic / hype → patterns below (stock openers, travel slop, taxonomy titles)
* wipe_reason → ``receipt_rejected`` only for stubs; ``generic_rejected`` for these
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence, Tuple

MOTORSPORT_OPENER_RES: Tuple[str, ...] = (
    r"\bstart your engines?\b",
    r"\bfasten your seatbelts?\b",
    r"\bbuckle up\b",
    r"\bhit the (?:road|gas)\b",
)

# Shared travel / influencer slop used by both hydration wipe and M8 ranking.
SHARED_PROSE_CLICHE_RES: Tuple[str, ...] = (
    r"\bcruise (?:under|through|along)\b",
    r"\bvast skies\b",
    r"\bendless horizons?\b",
    r"\b(?:adventure|journey) awaits?\b",
    r"\bopen road\b",
    r"\bgood vibes\b",
    r"\bbreath(?:e|taking) (?:in )?(?:the )?freedom\b",
    r"\b(?:explore|discover) more\b",
    r"\bnature(?:'s)? (?:beauty|symphony)\b",
    r"\bjoin me\b",
    r"\blet's dive\b",
    r"\byou won't believe\b",
    r"\bexciting moments?\b",
    r"\bunbelievable moments?\b",
    r"\bridin'? dirty\b",
    r"\bvibes? only\b",
    r"\bembrace the chaos\b",
    r"\bhidden gem\b",
    r"\bscenic (?:vibes?|drive|views?)\b",
    r"\b(?:travel|highway|cloud) (?:vibes?|watching)\b",
    r"\bhigh[- ]energy,?\s+first[- ]person\s+dashcam\b",
    r"\bthe video is a\s+(?:high[- ]energy|tense|exciting)\b",
    r"\bfrom inside a moving vehicle\b",
    r"\bcapturing a tense and confrontational journey\b",
    r"\bdashcam recording from inside\b",
)

# Hydration-only: longer travel variants + pure taxonomy / mood titles.
HYDRATION_ONLY_RES: Tuple[str, ...] = (
    r"\bendless (?:horizons?|road|roads|highway|highways|sky|skies)(?:\b|\s+(?:ahead|await|beckons?))",
    r"\b(?:adventure|journey|destiny|moment|magic) (?:awaits?|unfolds?|begins?|calls?|beckons?)\b",
    r"\bopen road(?:\s+(?:odyssey|calls?|beckons?|symphony|dreams?|magic))?\b",
    r"\b(?:open|endless) road\b",
    r"\bopen road\s+(?:odyssey|symphony|dreams?|magic|calls?|adventures?)\b",
    r"\bscenic (?:vibes?|drive|views?|stop|route|roads?|beauty)\b",
    r"\b(?:travel|highway|cloud) (?:vibes?|watching|symphony|dreams?)\b",
    r"\bhighway (?:symphony|dreams?|magic|melody|odyssey|tales?|stories)\b",
    r"\b(?:colorful|vibrant|stunning|breathtaking) blooms?\b",
    r"\bblooms? (?:stun|meet|burst|dance)\b",
    r"\bblooming (?:roads?|highways?|paths?)\b",
    r"\b(?:purple|red|yellow|pink) blooms?\s+(?:stun|meet|burst|dance)\b",
    r"\bdesert sands?\b",
    r"\bnature(?:'s)? (?:beauty|symphony|call|magic|wonders?)\b",
    r"\b(?:watch|witness) (?:the road|the world|the sky|nature|magic) (?:transform|unfold|change)\b",
    r"\b(?:watch|witness) (?:serenity|magic|nature|beauty) (?:meet|meets) (?:motion|sky|road|nature)\b",
    r"\bserenity meet(?:s)? motion\b",
    r"\b(?:road|highway|drive|journey) ahead\.?\s*$",
    r"\b(?:where|when) (?:the )?road meets (?:the )?(?:sky|horizon|sunset|dreams?)\b",
    r"\b(?:tranquil|peaceful|serene) (?:drive|journey|road|moments?)\b",
    r"\b(?:road|highway) (?:tales?|stories|chronicles|poetry)\b",
    r"\b(?:every )?mile (?:tells a story|matters|counts)\b",
    r"\bjourney captured\b",
    r"\b(?:scenic|epic|legendary) (?:moments?|adventures?|stops?)\b",
    r"\b(?:unforgettable|magical|legendary) (?:journey|drive|ride|moments?)\b",
    r"\bon the open road\b",
    r"\b(?:roads?|highways?) less travel(?:l)?ed\b",
    r"\bwhere the road takes (?:me|us|you)\b",
    r"^(?:road|drive|journey|adventure|highway|moment|vibes?|cruise|escape)\.?$",
    r"^(?:nature|horizon|scenery|landscape|outdoors?|transport|mode of transport|"
    r"vehicle|car|highway|road|sky|clouds?|trees?|water|travel|lifestyle|"
    r"automotive|beautiful|aesthetic|vibes?)\.?$",
    r"^(?:blue|green|red|yellow|orange|purple|pink|black|white|gray|grey)\s+"
    r"(?:sky|skies|trees?|horizon|nature|scenery|road|car|vibes?)\.?$",
    r"\bmode of transport\b",
    r"\bnature (?:views?|vibes?|scenes?|shots?|beauty)\b",
    r"\b(?:blue|green|golden) (?:skies|horizons?)\b",
)

# Ranking-only: meta / LLM self-talk that hydration may not need.
RANKING_ONLY_RES: Tuple[str, ...] = (
    r"\bunlock(ed)?\b",
    r"\bsecret\b",
    r"\bcontent creator\b",
    r"\bas an ai\b",
    r"\bwatch the road transform\b",
    r"\bin this (?:raw )?authentic moment\b",
    r"\bchannel(?:ing)? (?:my|your) emotions?\b",
)

# Backtest scan extras (end-of-line hype).
BACKTEST_EXTRA_RES: Tuple[str, ...] = (
    r"\blet'?s (?:go|ride|roll)\b[!.,]?\s*$",
)

MOTORSPORT_BAN_EXAMPLES: Tuple[str, ...] = (
    "Start your engines",
    "Buckle up",
    "Fasten your seatbelts",
    "Hit the gas",
    "Hit the road",
)


def _compile(patterns: Sequence[str]) -> Tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


@lru_cache(maxsize=1)
def hydration_cliche_patterns() -> Tuple[re.Pattern[str], ...]:
    """Patterns for ``_is_generic_caption`` / title generic wipe."""
    seen: set[str] = set()
    ordered: List[str] = []
    for p in (*MOTORSPORT_OPENER_RES, *SHARED_PROSE_CLICHE_RES, *HYDRATION_ONLY_RES):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return _compile(ordered)


@lru_cache(maxsize=1)
def ranking_cliche_patterns() -> Tuple[re.Pattern[str], ...]:
    """Patterns for ``_penalize_generic``."""
    seen: set[str] = set()
    ordered: List[str] = []
    for p in (*MOTORSPORT_OPENER_RES, *SHARED_PROSE_CLICHE_RES, *RANKING_ONLY_RES):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return _compile(ordered)


@lru_cache(maxsize=1)
def motorsport_opener_patterns() -> Tuple[re.Pattern[str], ...]:
    return _compile(MOTORSPORT_OPENER_RES)


@lru_cache(maxsize=1)
def backtest_opener_patterns() -> Tuple[re.Pattern[str], ...]:
    return _compile((*MOTORSPORT_OPENER_RES, *BACKTEST_EXTRA_RES))


def matches_prose_cliche(text: str, *, ranking: bool = False) -> bool:
    pats = ranking_cliche_patterns() if ranking else hydration_cliche_patterns()
    t = text or ""
    return any(p.search(t) for p in pats)


def prose_cliche_hits(text: str, *, ranking: bool = False) -> List[str]:
    pats = ranking_cliche_patterns() if ranking else hydration_cliche_patterns()
    t = text or ""
    return [p.pattern for p in pats if p.search(t)]


def motorsport_opener_hits(text: str) -> List[str]:
    t = text or ""
    return [p.pattern for p in motorsport_opener_patterns() if p.search(t)]


def is_sole_motorsport_opener(text: str) -> bool:
    """True when the whole title/caption is basically a stock race opener.

    Used for quality-gate hard reject under persona prefs — stronger than the
    light ``_penalize_generic`` −8 hit that still allows grounded long lines
    containing the phrase.
    """
    t = (text or "").strip()
    if not t:
        return False
    if not motorsport_opener_hits(t):
        return False
    if len(t) <= 40:
        return True
    if len(t) <= 56 and not re.search(r"\d", t):
        rem = t
        for p in motorsport_opener_patterns():
            rem = p.sub(" ", rem)
        rem = re.sub(r"[^\w\s]", " ", rem)
        rem = re.sub(r"\s+", " ", rem).strip()
        return len(rem) < 12
    return False


def prompt_motorsport_ban_line() -> str:
    examples = ", ".join(f'"{e}"' for e in MOTORSPORT_BAN_EXAMPLES[:4])
    return (
        f"Ban stock motorsport openers that could front any driving clip: {examples} "
        "(and 'Hit the road') — open on evidenced place, speed sample, music, or driver instead."
    )


def interaction_motorsport_ban_rule(*, intensity: int) -> str:
    return (
        f"High heat ({intensity}/5): invent hooks from THIS clip's evidence — ban stock "
        "motorsport openers ('Start your engines', 'Buckle up', 'Fasten your seatbelts', "
        "'Hit the gas') and any opener that could front any dashcam. Lead with a real "
        "place/speed/music/driver token."
    )


__all__ = [
    "MOTORSPORT_OPENER_RES",
    "MOTORSPORT_BAN_EXAMPLES",
    "SHARED_PROSE_CLICHE_RES",
    "HYDRATION_ONLY_RES",
    "RANKING_ONLY_RES",
    "hydration_cliche_patterns",
    "ranking_cliche_patterns",
    "motorsport_opener_patterns",
    "backtest_opener_patterns",
    "matches_prose_cliche",
    "prose_cliche_hits",
    "motorsport_opener_hits",
    "is_sole_motorsport_opener",
    "prompt_motorsport_ban_line",
    "interaction_motorsport_ban_rule",
]
