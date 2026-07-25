"""Canonical content identity across every understanding provider.

One artifact (``content_identity_v1``) answers "what IS this footage?" for
thumbnails, captions, and hashtags — instead of each call site re-running
keyword scans against hardcoded category lists.

Two layers:
  * Deterministic fusion (this module) — harvest open-vocabulary tokens from
    Twelve Labs / scene fusion prose, Google Vision, Video Intelligence,
    audio (music ID, YAMNet, transcript), and telemetry, then score
    cross-provider agreement. No keyword lists, no fixed taxonomy.
  * LLM resolution (``services.content_identity_llm``) — one structured call
    that names the subject in open vocabulary and ranks hero facts. Its
    output is merged back here through a grounding validator so ungrounded
    facts never reach prompts.

Same trust model as ``core.speed_consensus``: agreement across independent
providers is what earns confidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.speed_consensus import get_speed_consensus, scrub_untrusted_speed_claims

CONTENT_IDENTITY_ARTIFACT = "content_identity_v1"

# Hero-fact classes understood downstream (thumbnail headline selector, ML logging).
HERO_FACT_CLASSES: Tuple[str, ...] = (
    "entity",          # cross-provider agreed subject/object
    "landmark",        # Vision landmark
    "logo",            # Vision logo / brand
    "place",           # geo/telemetry place name
    "music",           # identified track/artist
    "on_screen_text",  # OCR
    "count",           # numeric quantity named by a provider
    "transcript",      # spoken line
    "speed",           # verified speed (consensus-gated)
    "sound",           # YAMNet sound class
)

_STOPWORDS: Set[str] = {
    "the", "and", "with", "from", "this", "that", "into", "over", "under",
    "then", "than", "very", "some", "more", "most", "each", "have", "has",
    "was", "are", "were", "been", "being", "will", "would", "could", "should",
    "video", "footage", "clip", "scene", "view", "shot", "shows", "showing",
    "there", "here", "their", "your", "what", "when", "where", "while",
    "during", "through", "along", "also", "just", "like", "onto", "them",
    "they", "these", "those", "much", "many", "such", "about",
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]{2,}")


def _content_words(text: str) -> Set[str]:
    """Lowercased content words (len >= 3, minus stopwords) for fuzzy matching."""
    return {
        w for w in _WORD_RE.findall(str(text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _clean(text: Any, *, max_len: int = 120) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    return out[:max_len].strip()


def _first_sentence(text: str, *, max_len: int = 140) -> str:
    t = _clean(text, max_len=600)
    if not t:
        return ""
    m = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)
    return _clean(m[0], max_len=max_len)


# ============================================================
# Evidence harvest — provider-attributed open-vocabulary tokens
# ============================================================

def _add_token(
    tokens: List[Dict[str, str]],
    provider: str,
    kind: str,
    text: Any,
    *,
    max_len: int = 120,
) -> None:
    t = _clean(text, max_len=max_len)
    if t and len(t) >= 3:
        tokens.append({"provider": provider, "kind": kind, "text": t})


def _iter_str_items(value: Any, *, limit: int = 24) -> List[str]:
    out: List[str] = []
    if isinstance(value, list):
        for item in value[:limit]:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("description", "name", "text", "label", "entity"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v)
                        break
    elif isinstance(value, str) and value.strip():
        out.append(value)
    return out


def build_identity_evidence(ctx: Any) -> Dict[str, Any]:
    """Harvest provider-attributed tokens from every understanding source.

    Returns ``{"tokens": [{provider, kind, text}], "prose": {provider: text}}``.
    No keyword lists — everything is open vocabulary straight from providers.
    """
    tokens: List[Dict[str, str]] = []
    prose: Dict[str, str] = {}

    # ── Scene understanding (Twelve Labs or fusion floor) ────────────────
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict):
        scene_provider = "fusion" if str(vu.get("source") or "").lower() == "fusion" else "twelvelabs"
        scene_txt = str(vu.get("scene_description") or vu.get("description") or "").strip()
        if scene_txt:
            prose[scene_provider] = _clean(scene_txt, max_len=900)
            _add_token(tokens, scene_provider, "scene", _first_sentence(scene_txt), max_len=140)
        for key in ("title_suggestion", "summary"):
            v = vu.get(key)
            if isinstance(v, str) and v.strip():
                _add_token(tokens, scene_provider, "scene", v, max_len=140)

    # ── Google Vision ────────────────────────────────────────────────────
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        for key, kind in (
            ("landmark_names", "landmark"),
            ("logo_names", "logo"),
            ("web_entities", "entity"),
            ("label_names", "label"),
            ("labels", "label"),
            ("objects", "object"),
        ):
            for item in _iter_str_items(vc.get(key)):
                _add_token(tokens, "vision", kind, item, max_len=80)
        ocr = str(vc.get("ocr_text") or "").strip()
        if ocr:
            prose["vision_ocr"] = _clean(ocr, max_len=500)
            _add_token(tokens, "vision", "on_screen_text", _first_sentence(ocr, max_len=100))

    # ── Video Intelligence ───────────────────────────────────────────────
    vi = (
        getattr(ctx, "video_intelligence_context", None)
        or getattr(ctx, "video_intelligence", None)
        or {}
    )
    if isinstance(vi, dict) and not vi.get("error"):
        for key, kind in (
            ("labels", "label"),
            ("label_names", "label"),
            ("shot_labels", "label"),
            ("objects", "object"),
            ("logos", "logo"),
        ):
            for item in _iter_str_items(vi.get(key)):
                _add_token(tokens, "video_intelligence", kind, item, max_len=80)
        vi_ocr = vi.get("ocr_text")
        if isinstance(vi_ocr, str) and vi_ocr.strip():
            _add_token(tokens, "video_intelligence", "on_screen_text", _first_sentence(vi_ocr, max_len=100))

    # ── Audio: music ID, YAMNet, keywords ────────────────────────────────
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        music = " — ".join(
            str(x) for x in (
                ac.get("music_artist") or ac.get("artist"),
                ac.get("music_title") or ac.get("track_title") or ac.get("title"),
            ) if x
        )
        if music:
            _add_token(tokens, "audio", "music", music, max_len=100)
        for item in _iter_str_items(ac.get("yamnet_events"), limit=8):
            _add_token(tokens, "audio", "sound", item, max_len=60)
        top_sound = ac.get("top_sound_class")
        if isinstance(top_sound, str) and top_sound.strip():
            _add_token(tokens, "audio", "sound", top_sound, max_len=60)
        for item in _iter_str_items(ac.get("suggested_keywords"), limit=12):
            _add_token(tokens, "audio", "label", item, max_len=60)

    # ── Speech transcript ────────────────────────────────────────────────
    transcript = str(getattr(ctx, "ai_transcript", "") or "").strip()
    if not transcript and isinstance(ac, dict):
        transcript = str(ac.get("transcript") or "").strip()
    if transcript:
        prose["transcript"] = _clean(transcript, max_len=600)
        _add_token(tokens, "speech", "transcript", _first_sentence(transcript, max_len=140))

    # ── User hints (title/caption/filename) — user-provided, so legit ────
    for attr in ("title", "caption"):
        v = getattr(ctx, attr, None)
        if isinstance(v, str) and v.strip() and _content_words(v):
            _add_token(tokens, "user", "hint", v, max_len=120)
    fname = str(getattr(ctx, "filename", "") or "")
    fname_base = re.sub(r"\.[a-z0-9]{2,4}$", "", fname, flags=re.IGNORECASE)
    fname_base = re.sub(r"[_\-.]+", " ", fname_base)
    if _content_words(fname_base):
        _add_token(tokens, "user", "hint", fname_base, max_len=80)

    # ── Telemetry / OSD geo (place facts, motion presence) ───────────────
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    if tel is not None:
        for attr in (
            "location_display", "location_road", "location_city",
            "location_state", "gazetteer_place_name", "padus_unit_name",
        ):
            v = getattr(tel, attr, None)
            if isinstance(v, str) and v.strip():
                _add_token(tokens, "telemetry", "place", v, max_len=80)
    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    if isinstance(osd, dict) and osd and not osd.get("skipped"):
        for key in ("location_display", "start_display", "place_name"):
            v = osd.get(key)
            if isinstance(v, str) and v.strip():
                _add_token(tokens, "osd", "place", v, max_len=80)

    return {"tokens": tokens, "prose": prose}


# ============================================================
# Cross-provider agreement
# ============================================================

def _group_tokens(tokens: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Group tokens naming the same thing across providers (fuzzy word overlap).

    Two tokens agree when they share at least one content word — open-vocab
    entity matching without any curated lists.
    """
    groups: List[Dict[str, Any]] = []
    for tok in tokens:
        words = _content_words(tok["text"])
        if not words:
            continue
        placed = False
        for g in groups:
            if words & g["words"]:
                g["words"] |= words
                g["providers"].add(tok["provider"])
                g["kinds"].add(tok["kind"])
                # Keep the shortest concrete label as the display text.
                if len(tok["text"]) < len(g["text"]) and tok["kind"] != "scene":
                    g["text"] = tok["text"]
                placed = True
                break
        if not placed:
            groups.append({
                "text": tok["text"],
                "words": set(words),
                "providers": {tok["provider"]},
                "kinds": {tok["kind"]},
            })
    return groups


_VERIFIED_KINDS = {"landmark", "logo", "music", "place"}


def _kind_to_fact_class(kinds: Set[str]) -> str:
    for k in ("landmark", "logo", "music", "place", "on_screen_text", "sound", "transcript"):
        if k in kinds:
            return k
    return "entity"


# ============================================================
# Deterministic identity
# ============================================================

def build_content_identity(ctx: Any, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic identity from cross-provider agreement (no LLM, no keywords)."""
    ev = evidence if isinstance(evidence, dict) else build_identity_evidence(ctx)
    tokens: List[Dict[str, str]] = list(ev.get("tokens") or [])
    prose: Dict[str, str] = dict(ev.get("prose") or {})

    consensus = get_speed_consensus(ctx)
    peak_mph = float(consensus.get("peak_mph") or 0)
    speed_conf = str(consensus.get("confidence") or "none")

    groups = _group_tokens(tokens)
    providers_seen = sorted({t["provider"] for t in tokens})

    def _group_score(g: Dict[str, Any]) -> float:
        score = 2.0 * len(g["providers"])
        if g["kinds"] & _VERIFIED_KINDS:
            score += 1.5
        if "scene" in g["kinds"]:
            score += 0.5
        return score

    ranked = sorted(groups, key=_group_score, reverse=True)

    hero_facts: List[Dict[str, Any]] = []
    for g in ranked[:8]:
        text = scrub_untrusted_speed_claims(g["text"], peak_mph)
        if not text:
            continue
        hero_facts.append({
            "text": text,
            "class": _kind_to_fact_class(g["kinds"]),
            "providers": sorted(g["providers"]),
            "score": round(_group_score(g), 2),
        })

    # Verified speed is a peer hero fact — only on high consensus confidence.
    if peak_mph >= 10 and speed_conf == "high":
        hero_facts.append({
            "text": f"{peak_mph:.0f} MPH peak (verified)",
            "class": "speed",
            "providers": sorted(set((consensus.get("agreeing") or [])) or {"telemetry"}),
            "score": 2.0 + len(consensus.get("agreeing") or []),
        })
    hero_facts.sort(key=lambda f: float(f.get("score") or 0), reverse=True)

    # Subject: scene first sentence when present, else top agreed entity.
    scene_txt = prose.get("twelvelabs") or prose.get("fusion") or ""
    subject = _first_sentence(scene_txt) if scene_txt else ""
    if not subject and hero_facts:
        subject = hero_facts[0]["text"]
    subject = scrub_untrusted_speed_claims(subject, peak_mph)

    multi_provider = [g for g in groups if len(g["providers"]) >= 2]
    if not tokens:
        confidence = "none"
    elif multi_provider or any(g["kinds"] & _VERIFIED_KINDS for g in groups):
        confidence = "high" if multi_provider and len(providers_seen) >= 2 else "medium"
    elif len(providers_seen) >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    do_not_invent: List[str] = []
    if peak_mph >= 10 and speed_conf in ("high", "medium"):
        do_not_invent.append(f"the only publishable speed is {peak_mph:.0f} MPH")
    else:
        do_not_invent.append("no verified speed data — never state a speed")
    vc = getattr(ctx, "vision_context", None) or {}
    try:
        faces = int(vc.get("face_count") or 0) if isinstance(vc, dict) else 0
    except (TypeError, ValueError):
        faces = 0
    if faces == 0:
        do_not_invent.append("no visible faces detected — do not invent or add people")
    ac = getattr(ctx, "audio_context", None) or {}
    has_music = bool(isinstance(ac, dict) and (ac.get("music_title") or ac.get("music_artist")))
    if not has_music:
        do_not_invent.append("no identified music track — do not name songs or artists")

    peak_metric_candidates: List[str] = []
    for f in hero_facts:
        if f["class"] in ("speed", "count") or re.search(r"\d", f["text"]):
            peak_metric_candidates.append(f["text"])
    peak_metric_candidates = peak_metric_candidates[:4]

    # Sensor-derived domain inference (GPS speed = a vehicle in motion) —
    # deterministic physics, not keyword scanning. LLM tags override on merge.
    domain_tags: List[Dict[str, Any]] = []
    if peak_mph >= 10 and speed_conf in ("high", "medium"):
        domain_tags.append({"tag": "automotive", "confidence": 0.7})

    return {
        "version": 1,
        "subject": subject,
        "activity": "",
        "setting": "",
        "domain_tags": domain_tags,  # open vocabulary — LLM resolution refines
        "hero_facts": hero_facts,
        "peak_metric_candidates": peak_metric_candidates,
        "do_not_invent": do_not_invent,
        "confidence": confidence,
        "novel_content": not multi_provider,
        "resolver": "deterministic",
        "providers_seen": providers_seen,
    }


# ============================================================
# LLM merge + grounding validation
# ============================================================

def _evidence_corpus_words(evidence: Dict[str, Any]) -> Set[str]:
    words: Set[str] = set()
    for tok in evidence.get("tokens") or []:
        words |= _content_words(tok.get("text") or "")
    for text in (evidence.get("prose") or {}).values():
        words |= _content_words(text)
    return words


def fact_is_grounded(text: str, corpus_words: Set[str]) -> bool:
    """A fact is grounded when it shares at least one content word with evidence."""
    fw = _content_words(text)
    return bool(fw and (fw & corpus_words))


def merge_llm_identity(
    base: Dict[str, Any],
    llm: Optional[Dict[str, Any]],
    *,
    evidence: Dict[str, Any],
    speed_consensus: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge the LLM resolution into the deterministic identity.

    Grounding contract: every LLM fact must share vocabulary with harvested
    evidence; speed claims are scrubbed against the consensus; ungrounded
    output is dropped silently (deterministic identity survives).
    """
    out = dict(base)
    if not isinstance(llm, dict) or not llm:
        return out

    corpus = _evidence_corpus_words(evidence)
    sc = speed_consensus or {}
    peak_mph = float(sc.get("peak_mph") or 0)
    speed_conf = str(sc.get("confidence") or "none")

    def _grounded_text(value: Any, *, max_len: int = 140) -> str:
        t = scrub_untrusted_speed_claims(_clean(value, max_len=max_len), peak_mph)
        return t if t and fact_is_grounded(t, corpus) else ""

    for key in ("subject", "activity", "setting"):
        t = _grounded_text(llm.get(key))
        if t:
            out[key] = t

    tags: List[Dict[str, Any]] = []
    for item in (llm.get("domain_tags") or [])[:5]:
        if isinstance(item, dict):
            tag = _clean(item.get("tag"), max_len=40).lower()
            try:
                conf = max(0.0, min(1.0, float(item.get("confidence") or 0)))
            except (TypeError, ValueError):
                conf = 0.0
        else:
            tag, conf = _clean(item, max_len=40).lower(), 0.5
        if tag:
            tags.append({"tag": tag, "confidence": round(conf, 2)})
    if tags:
        out["domain_tags"] = tags

    llm_facts: List[Dict[str, Any]] = []
    for item in (llm.get("hero_facts") or [])[:8]:
        if not isinstance(item, dict):
            continue
        text = _grounded_text(item.get("text"), max_len=120)
        if not text:
            continue
        fact_class = _clean(item.get("class"), max_len=20).lower()
        if fact_class not in HERO_FACT_CLASSES:
            fact_class = "entity"
        # Speed facts are consensus-gated regardless of what the LLM says.
        if fact_class == "speed" and not (peak_mph >= 10 and speed_conf == "high"):
            continue
        llm_facts.append({
            "text": text,
            "class": fact_class,
            "providers": sorted(
                str(p) for p in (item.get("providers") or []) if isinstance(p, str)
            ) or ["llm"],
            "score": 3.0,
        })
    if llm_facts:
        seen_words: List[Set[str]] = []
        merged: List[Dict[str, Any]] = []
        for f in llm_facts + list(out.get("hero_facts") or []):
            fw = _content_words(f["text"])
            if any(fw and fw & sw for sw in seen_words):
                continue
            seen_words.append(fw)
            merged.append(f)
        out["hero_facts"] = merged[:8]

    metrics = [
        _grounded_text(m, max_len=60)
        for m in (llm.get("peak_metric_candidates") or [])[:4]
    ]
    metrics = [m for m in metrics if m]
    if metrics:
        out["peak_metric_candidates"] = metrics

    extra_dni = [
        _clean(d, max_len=120)
        for d in (llm.get("do_not_invent") or [])[:4]
        if isinstance(d, str) and d.strip()
    ]
    if extra_dni:
        existing = {d.lower() for d in out.get("do_not_invent") or []}
        out["do_not_invent"] = list(out.get("do_not_invent") or []) + [
            d for d in extra_dni if d.lower() not in existing
        ]

    if isinstance(llm.get("novel_content"), bool):
        out["novel_content"] = llm["novel_content"]
    if out.get("confidence") in ("none", "low") and (tags or llm_facts):
        out["confidence"] = "medium"
    out["resolver"] = "llm+deterministic"
    return out


# ============================================================
# Cached accessor (same shape as get_speed_consensus)
# ============================================================

def get_content_identity(ctx: Any) -> Dict[str, Any]:
    """Return the persisted ``content_identity_v1`` artifact, building + caching once.

    Deterministic-only when the worker LLM resolution hasn't run (or failed) —
    consumers never need to care which layer produced it.
    """
    arts = getattr(ctx, "output_artifacts", None)
    if isinstance(arts, dict):
        existing = arts.get(CONTENT_IDENTITY_ARTIFACT)
        if isinstance(existing, dict) and existing.get("version"):
            return existing
    identity = build_content_identity(ctx)
    if isinstance(arts, dict):
        arts[CONTENT_IDENTITY_ARTIFACT] = identity
    return identity


def soft_bucket_for_identity(
    identity: Dict[str, Any],
    bucket_texts: Dict[str, str],
    *,
    default: str = "general",
    min_score: float = 2.0,
) -> str:
    """Soft-map an open-vocabulary identity onto legacy layout bucket keys.

    Used ONLY for layout/analytics compatibility (frame-selection prose,
    Studio layout chips, ML grouping) — prompts always consume the identity
    descriptor itself, never the bucket. Content that maps nowhere stays on
    ``default`` (the general prompt asks the model to identify from frames).
    """
    if not isinstance(identity, dict) or not bucket_texts:
        return default

    tags: List[str] = []
    for t in identity.get("domain_tags") or []:
        tag = str(t.get("tag") if isinstance(t, dict) else t or "").strip().lower()
        tag = re.sub(r"[\s\-]+", "_", tag)
        if tag:
            tags.append(tag)

    # Direct tag ↔ bucket-key match wins outright ("gardening" → gardening).
    for tag in tags:
        for key in bucket_texts:
            if key == default:
                continue
            k = key.lower()
            if tag == k or (len(tag) >= 4 and tag in k) or (len(k) >= 4 and k in tag):
                return key

    # Fuzzy fallback: weighted identity words vs bucket name + prose.
    weighted: Dict[str, float] = {}

    def _acc(text: Any, w: float) -> None:
        for word in _content_words(str(text or "")):
            weighted[word] = max(weighted.get(word, 0.0), w)

    for tag in tags:
        _acc(tag.replace("_", " "), 3.0)
    _acc(identity.get("subject"), 2.0)
    _acc(identity.get("activity"), 2.0)
    _acc(identity.get("setting"), 1.5)
    for f in (identity.get("hero_facts") or [])[:5]:
        if isinstance(f, dict):
            _acc(f.get("text"), 1.0)
    if not weighted:
        return default

    def _stem(w: str) -> str:
        return w[:-1] if w.endswith("s") and len(w) > 3 else w

    best_key, best_score = default, 0.0
    for key, text in bucket_texts.items():
        if key == default:
            continue
        bucket_words = {_stem(w) for w in _content_words(f"{key.replace('_', ' ')} {text}")}
        score = sum(w for word, w in weighted.items() if _stem(word) in bucket_words)
        if score > best_score:
            best_key, best_score = key, score
    return best_key if best_score >= min_score else default


def identity_scene_graph_view(identity: Dict[str, Any]) -> Dict[str, Any]:
    """Compact identity view for the M8 scene graph (prompt evidence)."""
    if not isinstance(identity, dict):
        return {}
    return {
        "subject": str(identity.get("subject") or "")[:140],
        "activity": str(identity.get("activity") or "")[:100],
        "setting": str(identity.get("setting") or "")[:80],
        "domain_tags": [
            {"tag": t.get("tag"), "confidence": t.get("confidence")}
            for t in (identity.get("domain_tags") or [])[:3]
            if isinstance(t, dict)
        ],
        "hero_facts": [
            {"text": f.get("text"), "class": f.get("class")}
            for f in (identity.get("hero_facts") or [])[:6]
            if isinstance(f, dict)
        ],
        "do_not_invent": list(identity.get("do_not_invent") or [])[:5],
        "confidence": identity.get("confidence"),
        "novel_content": bool(identity.get("novel_content")),
    }


def identity_prompt_line(identity: Dict[str, Any]) -> str:
    """One-line identity summary for template prompts (thumbnail brief vars)."""
    if not isinstance(identity, dict):
        return ""
    bits: List[str] = []
    if identity.get("subject"):
        bits.append(f"subject: {str(identity['subject'])[:120]}")
    if identity.get("activity"):
        bits.append(f"activity: {str(identity['activity'])[:80]}")
    tags = [
        str(t.get("tag") or "")
        for t in (identity.get("domain_tags") or [])[:3]
        if isinstance(t, dict) and t.get("tag")
    ]
    if tags:
        bits.append("domains: " + ", ".join(tags))
    facts = [
        str(f.get("text") or "")
        for f in (identity.get("hero_facts") or [])[:4]
        if isinstance(f, dict) and f.get("text")
    ]
    if facts:
        bits.append("hero facts: " + " | ".join(facts))
    dni = list(identity.get("do_not_invent") or [])[:3]
    if dni:
        bits.append("never invent: " + "; ".join(str(d) for d in dni))
    return " · ".join(bits)


def top_domain_tag(identity: Dict[str, Any]) -> str:
    tags = identity.get("domain_tags") or []
    if not tags:
        return ""
    best = max(tags, key=lambda t: float(t.get("confidence") or 0) if isinstance(t, dict) else 0)
    return str(best.get("tag") or "") if isinstance(best, dict) else ""


__all__ = [
    "CONTENT_IDENTITY_ARTIFACT",
    "HERO_FACT_CLASSES",
    "build_identity_evidence",
    "build_content_identity",
    "merge_llm_identity",
    "get_content_identity",
    "fact_is_grounded",
    "soft_bucket_for_identity",
    "identity_scene_graph_view",
    "identity_prompt_line",
    "top_domain_tag",
]
