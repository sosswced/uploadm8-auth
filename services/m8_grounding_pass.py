"""
M8 grounding pass 2 — claim↔evidence contract + deterministic critique.

Pass A (in m8_engine): model drafts captions (optionally with claims[]).
Pass B (this module): sanitize claims against an evidence catalog, strip
ungrounded sentences, and force MUST-USE tokens into winners when coverage
is too low. Hydration enforcer remains the last-resort gate.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def m8_grounding_pass2_enabled(user_settings: Optional[Dict[str, Any]] = None) -> bool:
    raw = (os.environ.get("M8_GROUNDING_PASS2") or "true").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    us = user_settings or {}
    if us.get("m8GroundingPass2") is not None:
        return bool(us.get("m8GroundingPass2"))
    if us.get("m8_grounding_pass2") is not None:
        return bool(us.get("m8_grounding_pass2"))
    return True


def build_evidence_catalog(
    scene_graph: Dict[str, Any],
    must_use: List[str],
    *,
    max_items: int = 24,
) -> Dict[str, Dict[str, Any]]:
    """Stable evidence_id → {text, lane} for claim binding."""
    catalog: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()

    def _add(lane: str, text: Any) -> None:
        s = str(text or "").strip()
        if not s or len(s) < 2:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        eid = f"e{len(catalog) + 1}"
        catalog[eid] = {"text": s[:160], "lane": lane}
        if len(catalog) >= max_items:
            return

    for tok in must_use or []:
        _add("must_use", tok)
        if len(catalog) >= max_items:
            return catalog

    geo = scene_graph.get("geo") or {}
    for k, lane in (
        ("road", "geo"),
        ("city", "geo"),
        ("state", "geo"),
        ("gazetteer_place", "geo"),
        ("protected_area_name", "geo"),
        ("display", "geo"),
    ):
        if geo.get(k):
            _add(lane, geo.get(k))

    vision = scene_graph.get("vision") or {}
    for lm in (vision.get("landmarks") or [])[:6]:
        _add("landmark", lm)
    for lg in (vision.get("logos") or [])[:4]:
        _add("logo", lg)

    pe = scene_graph.get("place_evidence") or {}
    if isinstance(pe, dict):
        for key, lane in (
            ("beaches", "beach"),
            ("monuments", "monument"),
            ("stadiums", "stadium"),
            ("sports_teams", "team"),
            ("license_plates", "plate"),
            ("places", "place"),
        ):
            for item in (pe.get(key) or [])[:4]:
                _add(lane, item)

    tr = scene_graph.get("transcript") or {}
    if isinstance(tr, dict) and tr.get("text"):
        phrase = str(tr.get("text") or "").strip().split(".")[0][:80]
        if phrase:
            _add("transcript", phrase)

    music = scene_graph.get("music") or {}
    if music.get("artist"):
        _add("music", music.get("artist"))
    if music.get("title"):
        _add("music", music.get("title"))

    return catalog


def _sanitize_claim_list(
    raw: Any,
    catalog: Dict[str, Dict[str, Any]],
    *,
    max_claims: int = 8,
) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    valid_ids = set(catalog.keys())
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()[:240]
        if not text:
            continue
        ids_raw = item.get("evidence_ids") or item.get("evidenceIds") or []
        if not isinstance(ids_raw, list):
            ids_raw = []
        ids = [str(x).strip() for x in ids_raw if str(x).strip() in valid_ids][:6]
        if not ids:
            # Try to bind by text overlap with catalog entries.
            blob = text.lower()
            for eid, meta in catalog.items():
                tok = str(meta.get("text") or "").lower()
                head = " ".join(tok.split()[:3])
                if head and head in blob:
                    ids.append(eid)
                if len(ids) >= 2:
                    break
        if not ids:
            continue
        try:
            conf = float(item.get("confidence") if item.get("confidence") is not None else 0.7)
        except (TypeError, ValueError):
            conf = 0.7
        conf = max(0.0, min(1.0, conf))
        out.append({"text": text, "evidence_ids": ids, "confidence": round(conf, 3)})
        if len(out) >= max_claims:
            break
    return out


def synthesize_claims_from_text(
    text: str,
    catalog: Dict[str, Dict[str, Any]],
    *,
    max_claims: int = 6,
) -> List[Dict[str, Any]]:
    """When the model omitted claims, derive them from catalog hits in the text."""
    blob = (text or "").lower()
    claims: List[Dict[str, Any]] = []
    for eid, meta in catalog.items():
        tok = str(meta.get("text") or "")
        head = " ".join(tok.lower().split()[:3])
        if head and head in blob:
            claims.append(
                {
                    "text": tok[:160],
                    "evidence_ids": [eid],
                    "confidence": 0.85,
                }
            )
        if len(claims) >= max_claims:
            break
    return claims


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


_FORMULA_STUB_RE = re.compile(
    # Checklist only: "88 MPH, Road" / "88 MPH · Place" — NOT "88 MPH — prose voice…"
    r"(?is)^\s*(?:anchored\s+in\s+)?\d{1,3}\s*mph\s*[,·]\s*.{0,80}\s*$"
)

# Compact timeline / hydration receipt — no audible persona.
_RECEIPT_THROUGH_RE = re.compile(
    r"(?is)^\s*\d{1,3}\s*mph\s+through\b.{0,90}$"
)
# Speed-less compact: "Through Livermore, CA — with A Boogie Wit da Hoodie"
_RECEIPT_THROUGH_GEO_RE = re.compile(
    r"(?is)^\s*through\s+[^!?]{2,60}\s*[—\-]\s*with\s+\S.{0,60}$"
)
_RECEIPT_RECORDED_RE = re.compile(
    r"(?is)^\s*\d{1,3}\s*mph\s+recorded\s+in\b"
)
_RECEIPT_CAPTURED_RE = re.compile(r"(?is)^\s*captured\s+at\b")
# Soft-inject remainder that is still just geo + optional "— with Artist".
# Kept for reference / callers; live stub logic uses _is_receipt_remainder().
_RECEIPT_REMAINDER_RE = re.compile(
    r"(?is)^\s*(?:through|near|on)\s+"
    r"[A-Za-z0-9][\w.'\-]*(?:\s+[A-Za-z0-9][\w.'\-]*){0,4}"
    r"(?:\s*,\s*[A-Za-z]{2,})?"
    r"(?:\s*[—\-]\s*with\s+[^!?]{2,40})?\s*$"
)
# "C Walker drives with 'Song' by Artist" fact weave after a speed lead.
_RECEIPT_DRIVES_WITH_RE = re.compile(
    r"(?is)^\s*\d{1,3}\s*mph\b.{0,40}\b(?:recorded\s+in|drives\s+with)\b"
)


def _is_receipt_remainder(rem: str) -> bool:
    """True when text after ``N MPH —`` is still a geo/music receipt, not voice.

    Compact receipts: short ``through/near/on Place`` or ``through Place — with Artist``.
    Spoken paraphrase that uses bare ``with`` (no em-dash music cue) is voice, not a stub.
    """
    r = (rem or "").strip()
    if not r:
        return True
    if " · " in r:
        return True
    # Music cue must use em/en dash before ``with`` (timeline receipt).
    if re.match(
        r"(?is)^\s*(?:through|near|on)\s+[^!?]{2,50}\s*[—\-]\s*with\s+[^!?]{2,40}\s*$",
        r,
    ):
        return True
    # Bare ``with`` without the dash music pattern → conversational prose.
    if re.search(r"(?i)\bwith\b", r):
        return False
    # Short geo-only remainder (no music, no extra clauses).
    if re.match(
        r"(?is)^\s*(?:through|near|on)\s+"
        r"[A-Za-z0-9][\w.'\-]*(?:\s+[A-Za-z0-9][\w.'\-]*){0,4}"
        r"(?:\s*,\s*[A-Za-z]{2,})?\s*$",
        r,
    ):
        return True
    return False


def is_formula_stub_caption(text: str) -> bool:
    """True for evidence *receipt templates* with no audible persona voice.

    Receipts (reject when Style/Tone/Voice prefs are set):
      - ``Anchored in 88 MPH, Garlock Road`` / ``110 MPH, Road`` / ``110 MPH · Place``
      - ``128 MPH through Allendale, CA — with iLoveMakonnen`` (compact timeline)
      - ``128 MPH recorded in Allendale…`` / ``Captured at 128 MPH…``

    Soft-inject titles ``154 MPH — <creative prose>`` are NOT stubs when the
    remainder is real voice (not ``through Place — with Artist``).
    """
    t = (text or "").strip()
    if not t:
        return False
    # Prefix receipts — flag even when the caption continues past 120 chars.
    if re.match(r"(?i)^\s*anchored\s+in\b", t):
        return True
    if _RECEIPT_CAPTURED_RE.match(t):
        return True
    if _RECEIPT_RECORDED_RE.match(t) or _RECEIPT_DRIVES_WITH_RE.match(t):
        return True

    # Em-dash / long hyphen soft-inject: creative remainder keeps; receipt remainder rejects.
    m = re.match(r"(?is)^\s*\d{1,3}\s*mph\s*[—\-]\s*(.+)$", t)
    if m:
        rem = m.group(1).strip()
        if " · " in t or _is_receipt_remainder(rem):
            return True
        if len(rem) >= 20:
            return False

    # Compact ``N MPH through Place…`` (with or without ``— with Artist``).
    head = t if len(t) <= 140 else t[:140]
    if _RECEIPT_THROUGH_RE.match(head):
        return True
    # Speed-less compact timeline: ``Through Place — with Artist``.
    if len(t) <= 120 and _RECEIPT_THROUGH_GEO_RE.match(t):
        return True

    if len(t) > 120:
        return False
    return bool(_FORMULA_STUB_RE.match(t))


def persona_voice_required(
    user_settings: Optional[Dict[str, Any]] = None,
    *,
    style_ui: str = "",
    tone_ui: str = "",
    voice_ui: str = "",
) -> bool:
    """True when Style/Tone/Voice ask for audible persona (not stock defaults).

    When true, compact timeline / Captured-at / recorded-in receipts must not
    ship as the final title or caption — prefer voice variants or voice_fallback.
    """
    try:
        from core.caption_creative import (
            DEFAULT_CAPTION_STYLE,
            DEFAULT_CAPTION_TONE,
            DEFAULT_CAPTION_VOICE,
            normalize_caption_style,
            normalize_caption_tone,
            normalize_caption_voice,
        )
    except Exception:
        return bool(style_ui or tone_ui or voice_ui)

    us = user_settings or {}
    s = normalize_caption_style(
        style_ui or us.get("captionStyle") or us.get("caption_style") or ""
    )
    t = normalize_caption_tone(
        tone_ui or us.get("captionTone") or us.get("caption_tone") or ""
    )
    v = normalize_caption_voice(
        voice_ui or us.get("captionVoice") or us.get("caption_voice") or ""
    )
    if s != DEFAULT_CAPTION_STYLE or t != DEFAULT_CAPTION_TONE or v != DEFAULT_CAPTION_VOICE:
        return True
    if us.get("captionCreativeResolvedFrom") or us.get("caption_creative_resolved_from"):
        return True
    if us.get("captionCreativeComboIndex") is not None or us.get(
        "caption_creative_combo_index"
    ) is not None:
        return True
    return False


def strip_ungrounded_sentences(
    text: str,
    claims: List[Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
) -> Tuple[str, int]:
    """
    Drop sentences that share no token with any claimed evidence text.
    Returns (new_text, stripped_count).

    Fail open on voice: if stripping would gut a longer caption into a thin
    stub, keep the original so persona/style/tone prose survives grounding.
    """
    if not text or not claims:
        return text, 0
    claim_blob = " ".join(str(c.get("text") or "") for c in claims).lower()
    for c in claims:
        for eid in c.get("evidence_ids") or []:
            meta = catalog.get(str(eid)) or {}
            claim_blob += " " + str(meta.get("text") or "").lower()
    kept: List[str] = []
    stripped = 0
    sentences = _sentence_split(text)
    for sent in sentences:
        tokens = {t for t in re.findall(r"[a-z0-9]{3,}", sent.lower())}
        if not tokens:
            kept.append(sent)
            continue
        # Keep if ≥1 content token overlaps claim/evidence blob.
        if any(t in claim_blob for t in tokens):
            kept.append(sent)
        else:
            stripped += 1
    if not kept:
        return text, 0
    joined = " ".join(kept)
    # Preserve persona voice: never reduce a real caption to a thin fact stub.
    if stripped and (
        len(joined) < 40
        or is_formula_stub_caption(joined)
        or (len(sentences) >= 2 and stripped >= max(1, (len(sentences) + 1) // 2) and len(text) >= 60)
    ):
        return text, 0
    return joined, stripped


def ensure_must_use_coverage(
    text: str,
    must_use: List[str],
    *,
    min_required: int = 2,
) -> Tuple[str, bool]:
    """Weave missing must_use facts into existing prose — never replace voice with a stub."""
    if not must_use:
        return text, False
    blob = (text or "").lower()
    hits = 0
    missing: List[str] = []
    for tok in must_use:
        head = " ".join(str(tok).lower().split()[:3])
        if head and head in blob:
            hits += 1
        else:
            missing.append(str(tok).strip())
    if hits >= min_required or not missing:
        return text, False
    need = max(0, min_required - hits)
    inject = ", ".join(missing[: max(1, need)])
    base = (text or "").strip()
    # Empty / already-a-stub: leave factual tokens only — no "Anchored in" brand
    # that the model then copies as the entire caption.
    if not base or is_formula_stub_caption(base):
        return inject.rstrip(" .") + ".", True
    # Soft-merge into existing voice (em dash closer, not a checklist sentence).
    core = base.rstrip(" .!?")
    return f"{core} — {inject}.", True


def apply_grounding_pass2_to_ranked(
    ranked: Dict[str, Any],
    scene_graph: Dict[str, Any],
    *,
    must_use: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Mutate ranked selection: attach catalog + claims, strip ungrounded prose,
    force must_use coverage on selected winners.
    """
    must_use = list(must_use or ranked.get("must_use") or [])
    catalog = build_evidence_catalog(scene_graph, must_use)
    report: Dict[str, Any] = {
        "enabled": True,
        "catalog_size": len(catalog),
        "platforms": {},
        "stripped_sentences": 0,
        "must_use_injected": 0,
        "claims_synthesized": 0,
    }

    platforms = ranked.get("platforms") or {}
    if not isinstance(platforms, dict):
        ranked["evidence_catalog"] = catalog
        ranked["grounding_pass2"] = report
        return ranked

    for pl, block in list(platforms.items()):
        if not isinstance(block, dict):
            continue
        selected = block.get("winner") or block.get("selected") or {}
        if not isinstance(selected, dict):
            continue
        caption = str(selected.get("caption") or "")
        title = str(selected.get("title") or "") if selected.get("title") is not None else ""
        claims = _sanitize_claim_list(selected.get("claims"), catalog)
        if not claims:
            claims = synthesize_claims_from_text(f"{title} {caption}", catalog)
            if claims:
                report["claims_synthesized"] += 1
        new_cap, n_strip = strip_ungrounded_sentences(caption, claims, catalog)
        report["stripped_sentences"] += n_strip
        new_cap, injected = ensure_must_use_coverage(new_cap, must_use)
        if injected:
            report["must_use_injected"] += 1

        variants = block.get("variants_ranked") or block.get("variants") or []
        # Voice salvage: if winner collapsed to a fact stub/receipt, prefer the
        # best non-stub variant and soft-weave must_use into that voice.
        voice_repaired = False
        if is_formula_stub_caption(new_cap) and isinstance(variants, list):
            for v in variants:
                if not isinstance(v, dict):
                    continue
                alt = str(v.get("caption") or "").strip()
                if not alt or is_formula_stub_caption(alt) or len(alt) < 40:
                    continue
                alt2, _ = ensure_must_use_coverage(alt, must_use)
                if alt2 and not is_formula_stub_caption(alt2):
                    new_cap = alt2
                    injected = True
                    voice_repaired = True
                    report["must_use_injected"] = int(report.get("must_use_injected") or 0) + 1
                    break

        # Never publish Anchored-in / · / through-Place receipts as the whole caption.
        if is_formula_stub_caption(new_cap):
            toks = [str(t).strip() for t in (must_use or []) if str(t).strip()][:3]
            if toks:
                if len(toks) == 1:
                    woven = f"Out here with {toks[0]} still in frame — what the clip actually shows."
                elif len(toks) == 2:
                    woven = f"Out here with {toks[0]} and {toks[1]} — what the clip actually shows."
                else:
                    woven = (
                        f"Out here with {toks[0]}, {toks[1]}, and {toks[2]} "
                        "— what the clip actually shows."
                    )
                new_cap = woven[:520]
                voice_repaired = True
                report["stub_replaced"] = int(report.get("stub_replaced") or 0) + 1

        # Title salvage: reject compact/receipt titles when a voice variant exists.
        new_title = title
        if is_formula_stub_caption(new_title) and isinstance(variants, list):
            for v in variants:
                if not isinstance(v, dict):
                    continue
                alt_t = str(v.get("title") or "").strip()
                if alt_t and not is_formula_stub_caption(alt_t) and len(alt_t) >= 12:
                    new_title = alt_t[:120]
                    voice_repaired = True
                    report["title_receipt_repaired"] = int(
                        report.get("title_receipt_repaired") or 0
                    ) + 1
                    break
            else:
                # Lift from repaired caption when title is still a receipt.
                first = re.split(r"(?<=[.!?])\s+", new_cap, maxsplit=1)[0].strip()
                if (
                    first
                    and len(first) >= 12
                    and not is_formula_stub_caption(first)
                ):
                    new_title = first[:120]
                    voice_repaired = True
                    report["title_from_caption"] = int(
                        report.get("title_from_caption") or 0
                    ) + 1

        selected = dict(selected)
        selected["caption"] = new_cap
        if new_title:
            selected["title"] = new_title[:120]
        elif selected.get("title") is not None and is_formula_stub_caption(
            str(selected.get("title") or "")
        ):
            # Drop receipt title rather than ship it; downstream will fill voice.
            selected["title"] = None
        selected["claims"] = claims
        block["winner"] = selected
        block["selected"] = selected  # alias for newer consumers

        if isinstance(variants, list):
            for v in variants:
                if not isinstance(v, dict):
                    continue
                v_claims = _sanitize_claim_list(v.get("claims"), catalog)
                if not v_claims:
                    v_claims = synthesize_claims_from_text(
                        f"{v.get('title') or ''} {v.get('caption') or ''}",
                        catalog,
                    )
                v["claims"] = v_claims

        report["platforms"][str(pl)] = {
            "claims": len(claims),
            "stripped": n_strip,
            "must_use_injected": injected,
            "voice_repaired": voice_repaired,
        }
        platforms[pl] = block

    ranked["platforms"] = platforms
    ranked["evidence_catalog"] = catalog
    ranked["grounding_pass2"] = report
    ranked["claims"] = {
        pl: ((platforms.get(pl) or {}).get("winner") or {}).get("claims") or []
        for pl in platforms
    }
    return ranked


def claims_prompt_section(catalog: Dict[str, Dict[str, Any]]) -> str:
    if not catalog:
        return ""
    lines = [f"  - {eid}: [{meta.get('lane')}] {meta.get('text')}" for eid, meta in list(catalog.items())[:20]]
    return f"""
EVIDENCE CATALOG (bind claims to these ids only):
{chr(10).join(lines)}

CLAIMS CONTRACT (required when catalog is non-empty):
- Each variant MUST include "claims": [ {{ "text": "...", "evidence_ids": ["e1", ...], "confidence": 0.0-1.0 }} ].
- Every factual noun phrase (speed, place, song, landmark) in caption/title should appear in some claim.text.
- evidence_ids MUST reference catalog ids above; never invent ids.
- Keep style/tone/voice prose — only omit sentences that invent false facts not in the catalog.
- Never write checklist stubs like "Anchored in 110 MPH, Road Name",
  "128 MPH through Place — with Artist", or "128 MPH recorded in Place…" as the
  whole caption/title; weave facts into Style/Tone/Voice prose.
"""


__all__ = [
    "m8_grounding_pass2_enabled",
    "build_evidence_catalog",
    "apply_grounding_pass2_to_ranked",
    "claims_prompt_section",
    "synthesize_claims_from_text",
    "ensure_must_use_coverage",
    "strip_ungrounded_sentences",
    "is_formula_stub_caption",
    "persona_voice_required",
]
