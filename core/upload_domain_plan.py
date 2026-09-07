"""Planned upload domains — turn running Vision / VI / web facts into titles.

We cannot enumerate every club, restaurant, or landmark. Google Vision, Video
Intelligence, and web-entity matching already name what is on screen. This
module pre-plans the *kinds* of uploads (sports, food, travel, cars, …) and
which service buckets to promote so any clip can auto-title from live analysis.

Kit recipes are optional boosters when Vision reports colors + sport labels
but has not yet spelled the club name.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.vision_labels import is_generic_vision_label

# Planned coverage for everything that commonly gets uploaded.
PLANNED_DOMAINS: Dict[str, Dict[str, Any]] = {
    "sports": {
        "label": "Sports / stadium / match",
        "title_buckets": ("web_matches", "places", "brands", "sports"),
        "detect": (
            "soccer", "football", "stadium", "arena", "jersey", "basketball",
            "baseball", "tennis", "hockey", "golf", "nfl", "nba", "mlb",
            "kickoff", "bleacher",
        ),
    },
    "automotive": {
        "label": "Driving / vehicles",
        "title_buckets": ("vehicles", "brands", "places", "signage", "web_matches"),
        "detect": ("car", "truck", "highway", "dashboard", "windshield", "mustang"),
    },
    "food": {
        "label": "Food / cooking",
        "title_buckets": ("food", "restaurants", "web_matches", "brands"),
        "detect": ("food", "cooking", "restaurant", "kitchen", "recipe", "plating"),
    },
    "travel": {
        "label": "Travel / outdoors / landmarks",
        "title_buckets": ("places", "web_matches", "outdoors", "landmarks"),
        "detect": ("landmark", "beach", "mountain", "park", "travel", "sunset"),
    },
    "music": {
        "label": "Music / concert / artist",
        "title_buckets": ("people", "brands", "web_matches", "places"),
        "detect": ("concert", "stage", "guitar", "dj", "festival", "microphone"),
    },
    "lifestyle": {
        "label": "Lifestyle / vlog / people",
        "title_buckets": ("people", "places", "web_matches", "brands"),
        "detect": ("person", "selfie", "vlog", "street"),
    },
    "tech": {
        "label": "Tech / product",
        "title_buckets": ("products", "brands", "web_matches", "text_on_screen"),
        "detect": ("phone", "laptop", "gadget", "unbox"),
    },
    "beauty": {
        "label": "Beauty / fashion",
        "title_buckets": ("products", "colors", "brands", "web_matches"),
        "detect": ("makeup", "fashion", "lipstick", "skincare"),
    },
    "gaming": {
        "label": "Gaming",
        "title_buckets": ("products", "brands", "text_on_screen", "web_matches"),
        "detect": ("game", "controller", "esport", "twitch"),
    },
    "education": {
        "label": "Education / how-to",
        "title_buckets": ("text_on_screen", "objects", "web_matches"),
        "detect": ("classroom", "whiteboard", "tutorial"),
    },
    "news": {
        "label": "News / commentary",
        "title_buckets": ("text_on_screen", "places", "web_matches", "people"),
        "detect": ("news", "headline", "interview"),
    },
    "fitness": {
        "label": "Fitness",
        "title_buckets": ("sports", "people", "web_matches"),
        "detect": ("gym", "workout", "yoga", "running"),
    },
    "general": {
        "label": "Auto (detected)",
        "title_buckets": ("web_matches", "places", "brands", "objects"),
        "detect": (),
    },
}

# Distinctive kits: Vision colors + sport labels → club name when web/OCR is quiet.
PLANNED_KITS: Tuple[Dict[str, Any], ...] = (
    {
        "display": "FC Barcelona",
        "domain": "sports",
        "colors": frozenset({"red", "blue", "navy", "maroon"}),
        "need_colors": (frozenset({"red", "maroon"}), frozenset({"blue", "navy"})),
        "need_any": ("soccer", "stadium", "jersey", "football field"),
        "block_if": ("psg", "paris saint", "real madrid"),
    },
)

_GENERIC_TITLE_SLUGS = frozenset(
    {
        "person", "people", "clothing", "night", "sky", "outdoor", "indoor",
        "sports", "stadium", "arena", "jersey", "soccer", "football",
        "vehicle", "car", "road", "building", "light", "darkness",
        "human", "man", "woman", "face", "hair", "shirt", "photography",
    }
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _is_title_worthy(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if len(raw) < 3 or len(raw) > 80:
        return False
    slug = _slug(raw)
    if not slug or slug in _GENERIC_TITLE_SLUGS:
        return False
    if is_generic_vision_label(raw):
        return False
    return True


def recognition_flat(ctx: Any) -> Dict[str, List[str]]:
    """Live service catalog from attach_visual_recognition, or build on the fly."""
    vr = getattr(ctx, "visual_recognition", None) or {}
    if isinstance(vr, dict):
        flat = vr.get("flat")
        if isinstance(flat, dict) and flat:
            return {str(k): list(v or []) for k, v in flat.items() if isinstance(v, list)}
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        flat = vc.get("recognition_flat")
        if isinstance(flat, dict) and flat:
            return {str(k): list(v or []) for k, v in flat.items() if isinstance(v, list)}
        try:
            from services.google_visual_recognition import build_recognition_catalog

            bundle = build_recognition_catalog(
                vision_context=vc,
                video_intelligence=getattr(ctx, "video_intelligence", None) or {},
                video_intelligence_context=getattr(ctx, "video_intelligence_context", None) or {},
                category=str(getattr(ctx, "thumbnail_category", None) or "general"),
                filename=str(getattr(ctx, "filename", "") or ""),
            )
            out = bundle.get("flat") or {}
            if isinstance(out, dict):
                return {str(k): list(v or []) for k, v in out.items() if isinstance(v, list)}
        except Exception:
            pass
    return {}


def detect_planned_domain(ctx: Any, *, flat: Optional[Dict[str, List[str]]] = None) -> str:
    """Pick a planned domain from user niche, then from Vision labels."""
    niche = str(getattr(ctx, "thumbnail_category", None) or "").strip().lower()
    if niche in PLANNED_DOMAINS and niche != "general":
        return niche
    vr = getattr(ctx, "visual_recognition", None) or {}
    if isinstance(vr, dict):
        n = str(vr.get("niche") or "").strip().lower()
        if n in PLANNED_DOMAINS and n != "general":
            return n
    blob_parts: List[str] = []
    data = flat if isinstance(flat, dict) else recognition_flat(ctx)
    for key in ("sports", "places", "brands", "food", "vehicles", "web_matches", "objects"):
        blob_parts.extend(str(x) for x in (data.get(key) or [])[:8])
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        blob_parts.extend(str(x) for x in (vc.get("label_names") or [])[:16])
        blob_parts.extend(str(x) for x in (vc.get("web_entities") or [])[:8])
    blob = " ".join(blob_parts).lower()
    best = "general"
    best_hits = 0
    for domain, spec in PLANNED_DOMAINS.items():
        if domain == "general":
            continue
        hits = sum(1 for tok in spec.get("detect") or () if tok in blob)
        if hits > best_hits:
            best = domain
            best_hits = hits
    return best


def service_named_titles(
    ctx: Any,
    *,
    domain: str = "",
    limit: int = 4,
) -> List[str]:
    """Proper nouns Vision / web / logos already returned — no team enum required."""
    flat = recognition_flat(ctx)
    domain = domain or detect_planned_domain(ctx, flat=flat)
    spec = PLANNED_DOMAINS.get(domain) or PLANNED_DOMAINS["general"]
    buckets: Sequence[str] = spec.get("title_buckets") or ("web_matches", "places", "brands")
    out: List[str] = []
    seen = set()
    for bucket in buckets:
        for raw in flat.get(bucket) or []:
            text = re.sub(r"\s+", " ", str(raw or "").strip())
            if not _is_title_worthy(text):
                continue
            key = _slug(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text[:80])
            if len(out) >= limit:
                return out
    # Raw Vision fields if catalog was never attached (tests / early stages).
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict) and len(out) < limit:
        extras: List[str] = []
        extras.extend(str(x) for x in (vc.get("web_entities") or [])[:8])
        extras.extend(str(x) for x in (vc.get("web_best_guess") or [])[:4])
        extras.extend(str(x) for x in (vc.get("landmark_names") or [])[:4])
        extras.extend(str(x) for x in (vc.get("logo_names") or [])[:4])
        for raw in extras:
            if isinstance(raw, dict):
                raw = raw.get("description") or raw.get("name") or ""
            text = re.sub(r"\s+", " ", str(raw or "").strip())
            if not _is_title_worthy(text):
                continue
            key = _slug(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text[:80])
            if len(out) >= limit:
                break
    return out


def match_planned_kit(ctx: Any, *, domain: str = "", colors: Optional[set] = None) -> str:
    """Optional kit booster from Vision colors + sport labels."""
    try:
        from core.sports_identity import dominant_color_names, detect_sport_kind
    except Exception:
        return ""
    color_set = set(colors or ()) or set(dominant_color_names(ctx))
    vc = getattr(ctx, "vision_context", None) or {}
    labels = []
    if isinstance(vc, dict):
        labels = [str(x) for x in (vc.get("label_names") or [])]
    blob = " ".join(labels).lower()
    sport = detect_sport_kind(blob, labels=labels)
    domain = domain or detect_planned_domain(ctx)
    for kit in PLANNED_KITS:
        if kit.get("domain") and kit["domain"] != domain and domain != "general":
            if sport != "soccer":
                continue
        if any(tok in blob for tok in kit.get("block_if") or ()):
            continue
        need_any = kit.get("need_any") or ()
        if need_any and not any(tok in blob for tok in need_any):
            continue
        pairs = kit.get("need_colors") or ()
        if pairs and not all(color_set & group for group in pairs):
            continue
        return str(kit.get("display") or "")
    return ""


def compose_service_title(ctx: Any) -> str:
    """Best title from live analysis services + planned domain/kit recipes."""
    flat = recognition_flat(ctx)
    domain = detect_planned_domain(ctx, flat=flat)
    named = service_named_titles(ctx, domain=domain, limit=3)
    kit = match_planned_kit(ctx, domain=domain)
    if kit and not any(kit.lower() in n.lower() or n.lower() in kit.lower() for n in named):
        named = [kit, *named]
    if not named:
        return ""
    # Prefer "Club at Venue" when two distinct service names exist.
    if len(named) >= 2:
        a, b = named[0], named[1]
        if _slug(a) != _slug(b) and " at " not in a.lower():
            return f"{a} at {b}"[:90]
    if domain == "sports" and named:
        head = named[0]
        if "match" not in head.lower() and " at " not in head.lower():
            # Keep venue-only or club-only; add "match" only for club-like names.
            if kit and head == kit:
                return f"{head} match"[:90]
        return head[:90]
    return named[0][:90]


__all__ = [
    "PLANNED_DOMAINS",
    "PLANNED_KITS",
    "recognition_flat",
    "detect_planned_domain",
    "service_named_titles",
    "match_planned_kit",
    "compose_service_title",
]
