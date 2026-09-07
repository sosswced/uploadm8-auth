"""Infer sport + team identity from Vision labels, colors, landmarks, and OCR.

Team names are not invented from a lone color. Kit colors (e.g. Barça red+blue)
only name a club when the clip is already a soccer/stadium scene. Explicit
tokens (Camp Nou, Barcelona, FCB) always win.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Famous venues that do not contain the word "Stadium".
_STADIUM_ALIASES: Tuple[Tuple[str, str, str], ...] = (
    ("camp nou", "Camp Nou", "FC Barcelona"),
    ("spotify camp nou", "Camp Nou", "FC Barcelona"),
    ("santiago bernabeu", "Santiago Bernabéu", "Real Madrid"),
    ("santiago bernabéu", "Santiago Bernabéu", "Real Madrid"),
    ("old trafford", "Old Trafford", "Manchester United"),
    ("anfield", "Anfield", "Liverpool"),
)

_TEAM_ALIASES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("fc barcelona", "barcelona", "barça", "barca", "blaugrana"), "FC Barcelona"),
    (("real madrid",), "Real Madrid"),
    (("manchester united", "man utd", "man united"), "Manchester United"),
    (("liverpool",), "Liverpool"),
    (("arsenal",), "Arsenal"),
    (("chelsea",), "Chelsea"),
    (("psg", "paris saint-germain", "paris saint germain"), "PSG"),
)

_SOCCER_RE = re.compile(
    r"\b("
    r"soccer|association\s+football|football\s+player|football\s+field|"
    r"soccer\s+ball|soccer\s+player|futbol|fútbol|kickoff|goalkeeper|"
    r"football\s+match|premier\s+league|la\s+liga"
    r")\b",
    re.I,
)
_AMERICAN_FOOTBALL_RE = re.compile(
    r"\b(american\s+football|nfl|touchdown|super\s*bowl|helmet|quarterback)\b",
    re.I,
)
_STADIUM_CTX_RE = re.compile(
    r"\b(stadium|arena|ballpark|bleacher|grandstand|sports\s+field|"
    r"soccer\s+field|football\s+field|jersey|kit|tunnel)\b",
    re.I,
)
_GENERIC_FOOTBALL_RE = re.compile(r"\bfootball\b", re.I)

_RED_NAMES = frozenset({"red", "maroon", "burgundy", "crimson"})
_BLUE_NAMES = frozenset({"blue", "navy", "teal"})


def _norm_blob(parts: Iterable[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(str(p or "") for p in parts if str(p or "").strip())).strip()


def _iter_str_items(value: Any, *, limit: int = 24) -> List[str]:
    out: List[str] = []
    if isinstance(value, str) and value.strip():
        out.append(value.strip())
    elif isinstance(value, list):
        for item in value[:limit]:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("description", "name", "text", "label", "entity"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
    return out


def _vision_text_parts(ctx: Any) -> List[str]:
    parts: List[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        vc = {}
    for key in (
        "label_names",
        "labels",
        "logo_names",
        "landmark_names",
        "web_entities",
        "web_best_guess",
        "objects",
        "ocr_text",
    ):
        parts.extend(_iter_str_items(vc.get(key)))
    vi = (
        getattr(ctx, "video_intelligence_context", None)
        or getattr(ctx, "video_intelligence", None)
        or {}
    )
    if isinstance(vi, dict):
        for key in ("labels", "label_names", "logos", "objects", "shot_labels"):
            parts.extend(_iter_str_items(vi.get(key)))
        parts.extend(_iter_str_items(vi.get("on_screen_text") or vi.get("text_detections")))
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        parts.append(str(ac.get("transcript") or ""))
        structured = ac.get("transcript_structured") or {}
        if isinstance(structured, dict):
            parts.extend(_iter_str_items(structured.get("topics")))
            ne = structured.get("named_entities") or {}
            if isinstance(ne, dict):
                parts.extend(_iter_str_items(ne.get("organizations")))
                parts.extend(_iter_str_items(ne.get("places")))
    parts.append(str(getattr(ctx, "ai_transcript", None) or ""))
    return parts


def dominant_color_names(ctx: Any) -> Set[str]:
    """Named + RGB-derived colors from Vision image properties."""
    names: Set[str] = set()
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        return names
    for prop in vc.get("dominant_colors") or []:
        if not isinstance(prop, dict):
            continue
        raw = str(prop.get("name") or "").strip().lower()
        if raw and not raw.startswith("rgb("):
            names.add(raw)
        rgb = prop.get("rgb") or []
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            try:
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            except (TypeError, ValueError):
                continue
            if r > 120 and r >= g + 25 and r >= b + 15:
                names.add("red")
            if b > 80 and b >= r + 10 and b >= g:
                names.add("blue")
            if r < 50 and g < 50 and b > 80:
                names.add("navy")
            if r > 80 and r > g + 20 and r > b and g < 80:
                names.add("maroon")
    return names


def detect_sport_kind(blob: str, *, labels: Iterable[str] = ()) -> str:
    """Return ``soccer`` / ``american_football`` / ``sports`` / ````."""
    text = _norm_blob([blob, *labels])
    if _AMERICAN_FOOTBALL_RE.search(text):
        return "american_football"
    if _SOCCER_RE.search(text):
        return "soccer"
    if _GENERIC_FOOTBALL_RE.search(text) and not _AMERICAN_FOOTBALL_RE.search(text):
        # Vision often labels soccer as "Football" in Europe.
        if _STADIUM_CTX_RE.search(text):
            return "soccer"
        return "sports"
    if _STADIUM_CTX_RE.search(text):
        return "sports"
    return ""


def _uniq(items: Iterable[str], *, limit: int = 6) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for raw in items:
        s = re.sub(r"\s+", " ", str(raw or "").strip())
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s[:80])
        if len(out) >= limit:
            break
    return out


def infer_sports_identity(ctx: Any) -> Dict[str, Any]:
    """Fuse live Vision/web names first; planned kits only fill gaps."""
    parts = _vision_text_parts(ctx)
    blob = _norm_blob(parts)
    blob_l = blob.lower()
    colors = dominant_color_names(ctx)
    sport = detect_sport_kind(blob_l, labels=parts)
    teams: List[str] = []
    stadiums: List[str] = []
    sources: List[str] = []

    # Service-named entities (web / landmark / logo) — not a team enum.
    try:
        from core.upload_domain_plan import (
            detect_planned_domain,
            match_planned_kit,
            service_named_titles,
        )

        domain = detect_planned_domain(ctx)
        named = service_named_titles(ctx, domain=domain, limit=6)
        for name in named:
            low = name.lower()
            if any(tok in low for tok in ("stadium", "arena", "camp nou", "field", "park")):
                stadiums.append(name)
                sources.append("vision_place")
            else:
                teams.append(name)
                sources.append("vision_web")
        kit = match_planned_kit(ctx, domain=domain, colors=colors)
        if kit and kit not in teams:
            teams.append(kit)
            sources.append("planned_kit")
    except Exception:
        named = []
        kit = ""

    for alias, stadium_name, team_name in _STADIUM_ALIASES:
        if alias in blob_l:
            stadiums.append(stadium_name)
            teams.append(team_name)
            sources.append("stadium_alias")
            if not sport:
                sport = "soccer"

    for aliases, team_name in _TEAM_ALIASES:
        if any(re.search(rf"\b{re.escape(a)}\b", blob_l) for a in aliases):
            teams.append(team_name)
            sources.append("name_token")
            if not sport:
                sport = "soccer"

    if sport:
        sources.append("sport_labels")
    if colors:
        sources.append("dominant_colors")

    kit_match = kit if kit else ""
    if not kit_match and "FC Barcelona" in teams:
        kit_match = "FC Barcelona"

    return {
        "sport_kind": sport,
        "sports_teams": _uniq(teams),
        "stadiums": _uniq(stadiums),
        "kit_colors": sorted(colors),
        "kit_match": kit_match,
        "sources": _uniq(sources, limit=8),
    }


def sports_story_clause(identity: Optional[Dict[str, Any]]) -> str:
    """One hydration-story sentence from inferred sport/team/colors."""
    if not identity:
        return ""
    sport = str(identity.get("sport_kind") or "").strip()
    teams = [str(t).strip() for t in (identity.get("sports_teams") or []) if str(t).strip()]
    stadiums = [str(s).strip() for s in (identity.get("stadiums") or []) if str(s).strip()]
    colors = [str(c).strip() for c in (identity.get("kit_colors") or []) if str(c).strip()]
    bits: List[str] = []
    if sport == "soccer":
        bits.append("soccer match")
    elif sport == "american_football":
        bits.append("American football")
    elif sport == "sports":
        bits.append("sports venue")
    if teams:
        bits.append("team " + ", ".join(teams[:2]))
    if stadiums:
        bits.append("at " + ", ".join(stadiums[:2]))
    if colors and (teams or sport == "soccer"):
        show = [c for c in colors if c in _RED_NAMES or c in _BLUE_NAMES]
        if show:
            bits.append("kit colors " + " and ".join(show[:3]))
    if not bits:
        return ""
    return "Sports scene: " + "; ".join(bits) + "."


def sports_title_phrase(identity: Optional[Dict[str, Any]]) -> str:
    """Short publishable title from sport/team — never a filename."""
    if not identity:
        return ""
    teams = [str(t).strip() for t in (identity.get("sports_teams") or []) if str(t).strip()]
    stadiums = [str(s).strip() for s in (identity.get("stadiums") or []) if str(s).strip()]
    sport = str(identity.get("sport_kind") or "").strip()
    if teams and stadiums:
        return f"{teams[0]} at {stadiums[0]}"
    if teams:
        return f"{teams[0]} match"
    if stadiums and sport == "soccer":
        return f"Soccer at {stadiums[0]}"
    if stadiums:
        return stadiums[0]
    if sport == "soccer":
        return "Soccer night"
    return ""


__all__ = [
    "dominant_color_names",
    "detect_sport_kind",
    "infer_sports_identity",
    "sports_story_clause",
    "sports_title_phrase",
]
