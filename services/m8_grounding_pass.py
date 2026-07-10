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


def strip_ungrounded_sentences(
    text: str,
    claims: List[Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
) -> Tuple[str, int]:
    """
    Drop sentences that share no token with any claimed evidence text.
    Returns (new_text, stripped_count).
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
    for sent in _sentence_split(text):
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
    return " ".join(kept), stripped


def ensure_must_use_coverage(
    text: str,
    must_use: List[str],
    *,
    min_required: int = 2,
) -> Tuple[str, bool]:
    """Append a short factual clause when must_use coverage is below min_required."""
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
    base = (text or "").rstrip()
    if base and not base.endswith((".", "!", "?")):
        base += "."
    clause = f" Anchored in {inject}."
    return (base + clause).strip(), True


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
        selected = dict(selected)
        selected["caption"] = new_cap
        selected["claims"] = claims
        block["winner"] = selected
        block["selected"] = selected  # alias for newer consumers

        variants = block.get("variants_ranked") or block.get("variants") or []
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
- Every factual noun phrase in caption/title should appear in some claim.text.
- evidence_ids MUST reference catalog ids above; never invent ids.
- If you cannot ground a sentence, omit it from the caption.
"""


__all__ = [
    "m8_grounding_pass2_enabled",
    "build_evidence_catalog",
    "apply_grounding_pass2_to_ranked",
    "claims_prompt_section",
    "synthesize_claims_from_text",
    "ensure_must_use_coverage",
    "strip_ungrounded_sentences",
]
