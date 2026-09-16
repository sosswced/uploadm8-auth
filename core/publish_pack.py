"""Canonical publish pack — one story for thumbnail, caption, and hashtags.

``publish_pack_v1`` is built after content identity + hydration and before
thumbnail/caption. Consumers must read this pack instead of inventing parallel
OCR/logo headlines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Set

PUBLISH_PACK_ARTIFACT = "publish_pack_v1"

# Classes allowed as pack hook / caption fuel (not necessarily on-image paint).
_PAINTABLE_CLASSES = frozenset({"speed", "place", "music", "landmark", "entity", "transcript"})
# On-image typography: composition-first — only verified speed hooks may paint.
_ON_IMAGE_PAINT_CLASSES = frozenset({"speed"})
# Never paint alone — roadside / web false positives (Jordan Kuwait Bank class).
_NON_PAINT_CLASSES = frozenset({"logo", "on_screen_text"})

_SPEED_PAINT_RE = re.compile(r"\b\d{1,3}\s*mph\b", re.IGNORECASE)
_LAYOUT_BAN_RE = re.compile(
    r"(?i)\b(?:two subjects|two faces|expressive faces|rivalry|reaction shot|"
    r"shock(?:ed)? face|lower[- ]third|bottom text|stacked text|bold text|"
    r"large bold|dual tension|text bias)\b"
)


def brand_safe_scene_spine(source: Mapping[str, Any]) -> str:
    """From-scratch scene line: energy and route vibe only. No brands, songs, or digits."""
    raw = str(source.get("pikzels_spine") or source.get("_uploadm8_pikzels_spine") or "")
    low = raw.lower()
    if "peak effort" in low:
        energy = "peak effort"
    elif "cruise" in low:
        energy = "cruise, not a speed peak"
    else:
        energy = "steady"
    geo = str(source.get("geo_context") or "").strip()
    geo = re.sub(r"(?i)\blocation\b", "", geo)
    geo = re.sub(r"(?i)\b\d{1,3}\s*mph\b", "", geo)
    geo = re.sub(r"(?i)\b(?:bank|shipping|tenente|filename|img_\d+)\b", "", geo)
    geo = re.sub(r"\s{2,}", " ", geo).strip(" ,.;")
    bits = ["driving energy" if ("dashcam" in low or "driving" in low) else "scene"]
    if geo and len(geo) > 2:
        bits.append(geo[:80])
    bits.append(energy)
    if source.get("music_context") or "music" in low:
        bits.append("music in the background")
    return "; ".join(bits)[:220]


def scrub_studio_layout_text(text: Any, *, dashcam: bool = False) -> str:
    """Drop layout phrases that invent faces or demand banners. Crop/color may remain."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if dashcam or _LAYOUT_BAN_RE.search(raw):
        if _LAYOUT_BAN_RE.search(raw) or re.search(r"(?i)\b(?:face|faces|subject|text)\b", raw):
            return ""
        raw = _LAYOUT_BAN_RE.sub("", raw)
    raw = re.sub(r"\s{2,}", " ", raw).strip(" .;,")
    return raw[:180]

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9']{2,}")
_STOP = frozenset(
    {
        "the", "and", "with", "from", "this", "that", "into", "over", "under",
        "video", "footage", "clip", "scene", "view", "shot", "shows", "showing",
        "driving", "drive", "near", "in", "on", "at", "for", "via",
    }
)


def _content_words(text: Any) -> Set[str]:
    return {
        w
        for w in _WORD_RE.findall(str(text or "").lower())
        if len(w) >= 3 and w not in _STOP
    }


def subject_token_overlap(headline: Any, subject: Any, *, min_overlap: int = 1) -> bool:
    """True when headline shares enough content words with identity subject."""
    hw = _content_words(headline)
    sw = _content_words(subject)
    if not hw or not sw:
        return False
    return len(hw & sw) >= min_overlap


def headline_agrees_with_pack(headline: Any, pack: Optional[Dict[str, Any]]) -> bool:
    """True when headline matches pack hook or overlaps pack subject."""
    if not isinstance(pack, dict):
        return False
    from core.thumbnail_text import thumbnail_headline_body

    h_body = thumbnail_headline_body(headline)
    if not h_body:
        return False
    hook = thumbnail_headline_body(pack.get("hook_line") or "")
    if hook and (h_body == hook or hook in h_body or h_body in hook):
        return True
    if subject_token_overlap(headline, pack.get("subject") or ""):
        return True
    # Seeds may reinforce geo/music hooks — never ambient logo slugs.
    ambient = {
        "jordankuwaitbank",
        "kuwaitbank",
        "realunited",
        "klankosova",
        "klankoso",
    }
    compact = h_body.replace(" ", "")
    for seed in pack.get("hashtag_seeds") or []:
        seed_body = thumbnail_headline_body(seed)
        if not seed_body or len(seed_body) < 4:
            continue
        if seed_body in ambient or seed_body.replace(" ", "") in ambient:
            continue
        if seed_body in h_body or seed_body.replace(" ", "") in compact:
            return True
    return False


def prefer_pack_title_if_generic(title: Any, pack: Optional[Dict[str, Any]]) -> str:
    """Fail-soft: when LLM title is generic/off-spine, prefer pack hook or subject.

    Returns empty string when the existing title should be kept.
    """
    from core.helpers import clip_at_word_boundary

    raw = str(title or "").strip()
    if not isinstance(pack, dict):
        return ""
    subject = str(pack.get("subject") or "").strip()
    hook = str(pack.get("hook_line") or "").strip()
    if not subject and not hook:
        return ""
    low = raw.lower()
    generic_openers = (
        "pov:",
        "pov ",
        "wait for",
        "wait until",
        "this is why",
        "nobody expected",
        "you need to see",
        "you won't believe",
        "watch this",
        "omg",
        "untitled",
        "new video",
        "my video",
        "check this",
    )
    looks_generic = (not raw) or any(low.startswith(g) or f" {g}" in low for g in generic_openers)
    agrees = subject_token_overlap(raw, subject) or (
        hook and (hook.lower() in low or any(w in low for w in _content_words(hook)))
    )
    if agrees and not looks_generic:
        return ""
    # Build rarest concrete title: place+music-ish from subject, else hook, else subject.
    if hook and subject and not subject_token_overlap(hook, subject):
        candidate = f"{hook} — {subject}"
    elif hook:
        candidate = hook if not subject else (f"{subject}: {hook}" if len(subject) < 60 else hook)
    else:
        candidate = subject
    out = clip_at_word_boundary(candidate, 100)
    if not out or out.lower() == low:
        return ""
    return out


def is_paintable_pack_headline(
    headline: Any,
    pack: Optional[Dict[str, Any]],
    *,
    fact_class: str = "",
) -> bool:
    """Gate for Pikzels/PIL on-image text.

    Composition-first policy:
      - paint_policy ``none`` → never
      - location / business / filename / LOCATION banners → never
      - only earned numeric speed hooks (e.g. ``78 MPH``) may paint
      - logo / on_screen_text alone → never unless agrees with subject/hook
      - otherwise requires pack agreement (subject/hook) when pack exists
    """
    from core.thumbnail_text import (
        is_location_banner_headline,
        is_unusable_thumbnail_headline,
        thumbnail_headline_body,
    )

    raw = str(headline or "").strip()
    if not raw or is_unusable_thumbnail_headline(raw):
        return False
    if is_location_banner_headline(raw):
        return False
    pack = pack if isinstance(pack, dict) else {}
    policy = str(pack.get("paint_policy") or "none").strip().lower()
    if policy in {"none", "no_text", "no-text"}:
        return False
    cls = str(fact_class or "").strip().lower()
    if not cls and isinstance(pack, dict):
        cls = str(pack.get("hook_class") or "").strip().lower()
    # Only speed-class (or explicit MPH pattern) may become on-image typography.
    if cls and cls not in _ON_IMAGE_PAINT_CLASSES:
        return False
    if not _SPEED_PAINT_RE.search(raw):
        return False
    body = thumbnail_headline_body(raw)
    # Hard reject known ambient / false-logo cover phrases even without class.
    if body and any(
        bad in body.replace(" ", "")
        for bad in (
            "jordankuwaitbank",
            "kuwaitbank",
            "realunited",
            "klankosova",
            "klankoso",
            "mediterraneanshipping",
            "msc",
        )
    ):
        if not headline_agrees_with_pack(raw, pack):
            return False
    if cls in _NON_PAINT_CLASSES and not headline_agrees_with_pack(raw, pack):
        return False
    if pack.get("subject") or pack.get("hook_line"):
        return headline_agrees_with_pack(raw, pack)
    if cls and cls not in _ON_IMAGE_PAINT_CLASSES:
        return False
    return True


def get_publish_pack(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        pack = arts.get(PUBLISH_PACK_ARTIFACT)
        if isinstance(pack, dict) and pack:
            return pack
    return {}


def _hook_from_speed(ctx: Any) -> str:
    try:
        from core.speed_consensus import publishable_peak_mph

        mph = publishable_peak_mph(ctx)
        if mph >= 10:
            return f"{mph:.0f} MPH"
    except Exception:
        pass
    return ""


def _hook_from_place(identity: Dict[str, Any], hydration: Dict[str, Any]) -> str:
    for fact in identity.get("hero_facts") or []:
        if isinstance(fact, dict) and fact.get("class") == "place":
            t = str(fact.get("text") or "").strip()
            if t:
                return t[:34]
    ev = hydration.get("evidence") if isinstance(hydration.get("evidence"), dict) else {}
    geo = ev.get("geo") if isinstance(ev, dict) else {}
    if isinstance(geo, dict):
        for key in ("display", "city", "road"):
            v = str(geo.get(key) or "").strip()
            if v and len(v) >= 3:
                return v[:34]
    return ""


def _hook_from_music(identity: Dict[str, Any], hydration: Dict[str, Any]) -> str:
    for fact in identity.get("hero_facts") or []:
        if isinstance(fact, dict) and fact.get("class") == "music":
            t = str(fact.get("text") or "").strip()
            if t:
                return t[:34]
    ev = hydration.get("evidence") if isinstance(hydration.get("evidence"), dict) else {}
    music = ev.get("music") if isinstance(ev, dict) else {}
    if isinstance(music, dict):
        artist = str(music.get("artist") or "").strip()
        title = str(music.get("title") or "").strip()
        if artist and title:
            return f"{artist}"[:34]
        return (artist or title)[:34]
    return ""


def _hashtag_seeds_from_pack_parts(
    *,
    subject: str,
    hook: str,
    identity: Dict[str, Any],
    hydration: Dict[str, Any],
) -> List[str]:
    from core.helpers import sanitize_hashtag_body

    seeds: List[str] = []
    seen: Set[str] = set()

    def _add(raw: Any) -> None:
        body = sanitize_hashtag_body(str(raw or ""), max_len=30)
        if not body or body in seen or len(body) < 3:
            return
        seen.add(body)
        seeds.append(body)

    for fact in (identity.get("hero_facts") or [])[:8]:
        if not isinstance(fact, dict):
            continue
        cls = str(fact.get("class") or "")
        if cls in _NON_PAINT_CLASSES:
            continue
        _add(fact.get("text"))
    allow_energy = {"sendit", "highwayheat", "tripledigits", "speeddemon"}
    subject_words = _content_words(subject) | _content_words(hook)
    ambient: Set[str] = set()
    try:
        from services.hydration_enforcer import _AMBIENT_LOGO_SLUGS, _is_ambient_logo

        ambient = set(_AMBIENT_LOGO_SLUGS)
    except Exception:
        _is_ambient_logo = lambda _t: False  # type: ignore
        ambient = {
            "jordankuwaitbank",
            "kuwaitbank",
            "realunited",
            "klankosova",
            "klankoso",
        }
    for tag in hydration.get("signal_hashtags") or []:
        t = str(tag or "").strip().lstrip("#").lower()
        if not t:
            continue
        if t in allow_energy:
            _add(t)
            continue
        if t in ambient or _is_ambient_logo(t):
            continue
        # Drop long brand-like seeds that don't overlap subject (jordankuwaitbank).
        if subject_words and len(t) > 8 and not any(w in t for w in subject_words):
            if not subject_token_overlap(t, subject):
                continue
        # Reject camera-dump style slugs.
        if re.search(r"\d{6,}", t) and re.search(r"(?:cam|img|vid|mov)", t):
            continue
        _add(t)
    # Final ambient scrub (hero-fact path can still sanitize into ambient slugs).
    seeds = [s for s in seeds if s not in ambient and not _is_ambient_logo(s)]
    return seeds[:12]


def build_publish_pack(ctx: Any) -> Dict[str, Any]:
    """Rule-based pack from identity + hydration. Fail-soft empty pack on error."""
    from core.content_identity import get_content_identity
    from core.thumbnail_text import clean_thumbnail_headline

    identity = get_content_identity(ctx) if ctx is not None else {}
    if not isinstance(identity, dict):
        identity = {}
    hp = getattr(ctx, "hydration_payload", None) or {}
    if not isinstance(hp, dict):
        arts = getattr(ctx, "output_artifacts", None) or {}
        hp = arts.get("hydration_payload") if isinstance(arts, dict) else {}
    if isinstance(hp, str):
        try:
            import json

            hp = json.loads(hp)
        except Exception:
            hp = {}
    if not isinstance(hp, dict):
        hp = {}

    subject = str(identity.get("subject") or "").strip()[:140]
    driving = False
    dashcam = False
    try:
        from core.driving_evidence import has_driving_evidence

        driving = bool(has_driving_evidence(ctx))
    except Exception:
        driving = False
    category = str(hp.get("category") or identity.get("soft_bucket") or "").strip().lower()
    # Avoid importing thumbnail_stage (circular). Dashcam ≈ driving + automotive/general.
    dashcam = bool(
        driving
        and category in {"", "automotive", "general", "travel", "dashcam"}
    )
    us = getattr(ctx, "user_settings", None) or {}
    if str(us.get("content_category") or us.get("contentCategory") or "").lower() == "dashcam":
        dashcam = True

    speed_hook = _hook_from_speed(ctx)
    place_hook = _hook_from_place(identity, hp)
    music_hook = _hook_from_music(identity, hp)

    # Hook line still carries the best story token for captions/hashtags, but
    # on-image paint is composition-first: only earned speed (MPH) may paint.
    hook = ""
    hook_class = ""
    if driving or dashcam or category == "automotive":
        # Prefer verified drive story over logos.
        if speed_hook:
            hook, hook_class = speed_hook, "speed"
        elif place_hook:
            hook, hook_class = place_hook, "place"
        elif music_hook:
            hook, hook_class = music_hook, "music"
    else:
        for fact in identity.get("hero_facts") or []:
            if not isinstance(fact, dict):
                continue
            cls = str(fact.get("class") or "")
            if cls in _NON_PAINT_CLASSES:
                continue
            text = clean_thumbnail_headline(fact.get("text"), max_words=5)
            if text and (not subject or subject_token_overlap(text, subject) or cls in _PAINTABLE_CLASSES):
                hook, hook_class = text, cls
                break
        if not hook:
            hook = place_hook or music_hook or speed_hook
            hook_class = "place" if place_hook else ("music" if music_hook else ("speed" if speed_hook else ""))

    if hook and subject and hook_class == "logo":
        hook, hook_class = "", ""

    # First ship: never paint. Speed/place/music stay caption fuel only.
    paint_policy = "none"

    caption_spine = str(hp.get("anchor_phrase") or "").strip()[:220]
    if not caption_spine and subject:
        caption_spine = subject

    seeds = _hashtag_seeds_from_pack_parts(
        subject=subject, hook=hook, identity=identity, hydration=hp
    )

    visual_brief = {
        "moment": "hero_or_vi_keyframe" if (driving or dashcam) else "sharpest_frame",
        "composition": "preserve_pov" if dashcam else "styled_unique",
        # Place/music/logo never become lower-third banners — AI composition only.
        "text": "none",
        "category": category or "general",
    }

    faces_allowed = True
    dni = identity.get("do_not_invent") if isinstance(identity.get("do_not_invent"), list) else []
    for line in dni:
        if "no visible faces" in str(line).lower():
            faces_allowed = False
            break
    kind = "dashcam pov" if dashcam else ("driving" if driving else (category or "general"))
    mph = 0.0
    try:
        from core.speed_consensus import publishable_peak_mph

        mph = float(publishable_peak_mph(ctx) or 0)
    except Exception:
        mph = 0.0
    if mph >= 50:
        energy = "peak effort"
    elif mph >= 10:
        energy = "cruise, not a speed peak"
    else:
        energy = "steady"
    audio = ""
    if music_hook and hook_class == "music":
        audio = "music present"
    elif music_hook:
        audio = "music in the background"
    spine_bits = [kind, subject[:80] if subject else "", energy]
    if audio:
        spine_bits.append(audio)
    pikzels_spine = "; ".join(b for b in spine_bits if b)[:280]

    pack: Dict[str, Any] = {
        "v": 1,
        "subject": subject,
        "hook_line": hook,
        "hook_class": hook_class,
        "caption_spine": caption_spine,
        "pikzels_spine": pikzels_spine,
        "faces_allowed": faces_allowed,
        "hashtag_seeds": seeds,
        "visual_brief": visual_brief,
        "paint_policy": paint_policy,
        "evidence_refs": {
            "identity_confidence": identity.get("confidence"),
            "identity_resolver": identity.get("resolver"),
            "hydration_category": category,
            "driving_evidence": driving,
            "dashcam_pov": dashcam,
        },
    }
    return pack


def attach_publish_pack(ctx: Any, pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build (if needed) and store pack on ctx.output_artifacts."""
    built = pack if isinstance(pack, dict) and pack else build_publish_pack(ctx)
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        arts = {}
        try:
            ctx.output_artifacts = arts
        except Exception:
            pass
    arts[PUBLISH_PACK_ARTIFACT] = built
    return built


__all__ = [
    "PUBLISH_PACK_ARTIFACT",
    "attach_publish_pack",
    "build_publish_pack",
    "get_publish_pack",
    "headline_agrees_with_pack",
    "is_paintable_pack_headline",
    "prefer_pack_title_if_generic",
    "brand_safe_scene_spine",
    "scrub_studio_layout_text",
    "subject_token_overlap",
]
