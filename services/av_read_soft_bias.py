"""
P4 soft packaging bias from AV pack / distill signals.

Flag: AV_READ_SOFT_BIAS (default OFF). Soft prompt / rank hints only —
never writes prefs, never bypasses identity or grounding.

Same-upload: when pack is not written yet (after identity, before hydration),
boost from content_identity hero class (source=identity_fallback).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8-worker")

_ALLOWED_TL = frozenset({"ok", "skipped", "failed", "disabled"})


def soft_bias_enabled() -> bool:
    try:
        from services.av_read_runtime_flags import flag_enabled

        return flag_enabled("AV_READ_SOFT_BIAS")
    except Exception:
        return (os.environ.get("AV_READ_SOFT_BIAS") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )


def _as_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _hero_class_from_identity(identity: Dict[str, Any]) -> str:
    for h in identity.get("hero_facts") or []:
        if isinstance(h, dict) and h.get("class"):
            return str(h.get("class")).strip().lower()[:32]
    return ""


def identity_soft_flags(arts: Any) -> Dict[str, Any]:
    """Flags from content_identity only (pack_present=0) for same-upload soft bias."""
    empty = {
        "pack_present": 0,
        "pack_tl_status": "na",
        "pack_needs_deep_teacher": 0,
        "pack_hero_class": "",
    }
    if not isinstance(arts, dict):
        return empty
    identity = _as_dict(arts.get("content_identity_v1"))
    hero = _hero_class_from_identity(identity)
    return {
        "pack_present": 0,
        "pack_tl_status": "na",
        "pack_needs_deep_teacher": 0,
        "pack_hero_class": hero,
    }


def coarse_pack_flags(arts: Any) -> Dict[str, Any]:
    """Coarse flags only — no free text (cardinality-safe for attribution)."""
    empty = {
        "pack_present": 0,
        "pack_tl_status": "na",
        "pack_needs_deep_teacher": 0,
        "pack_hero_class": "",
    }
    if not isinstance(arts, dict):
        return empty
    pack = arts.get("av_training_pack_v1")
    pack = _as_dict(pack) if not isinstance(pack, dict) else pack
    if not pack:
        meta = arts.get("av_training_pack_meta_v1")
        if isinstance(meta, dict):
            pack = meta
    if not isinstance(pack, dict) or not pack:
        return empty
    tl = str(pack.get("tl_status") or "").strip().lower()
    if tl not in _ALLOWED_TL:
        tl = "na"
    hero = ""
    for h in pack.get("hero_facts") or []:
        if isinstance(h, dict) and h.get("class"):
            hero = str(h.get("class")).strip().lower()[:32]
            break
    if not hero:
        hero = str(pack.get("identity_hero_fact_class") or "").strip().lower()[:32]
    if not hero:
        hero = _hero_class_from_identity(_as_dict(arts.get("content_identity_v1")))
    return {
        "pack_present": 1,
        "pack_tl_status": tl,
        "pack_needs_deep_teacher": 1 if pack.get("needs_deep_teacher") else 0,
        "pack_hero_class": hero,
    }


def resolve_soft_flags(arts: Any) -> Tuple[Dict[str, Any], str]:
    """
    Prefer pack flags; else identity fallback for same-upload soft bias.
    Returns (flags, source) where source is pack | identity_fallback | none.
    """
    pack_flags = coarse_pack_flags(arts)
    if pack_flags.get("pack_present") and pack_flags.get("pack_hero_class"):
        return pack_flags, "pack"
    if pack_flags.get("pack_present"):
        return pack_flags, "pack"
    id_flags = identity_soft_flags(arts)
    if id_flags.get("pack_hero_class"):
        return id_flags, "identity_fallback"
    return id_flags, "none"


def soft_strategy_block(ctx: Any) -> str:
    """One soft prompt block for m8_strategy_context (empty if flag off / no signal)."""
    if not soft_bias_enabled():
        return ""
    arts = getattr(ctx, "output_artifacts", None) or {}
    flags, source = resolve_soft_flags(arts)
    hc = flags.get("pack_hero_class") or ""
    if not hc and not flags.get("pack_present"):
        return ""
    bits: List[str] = ["AV READ SOFT BIAS (advisory; grounding still judges):"]
    if source == "identity_fallback":
        bits.append("(identity fallback — pack not written yet)")
    if flags.get("pack_needs_deep_teacher"):
        bits.append("prefer denser evidence / avoid inventing detail when teacher was deep.")
    if hc:
        bits.append(f"lean packaging toward hero class “{hc}” when evidence agrees.")
    tl = flags.get("pack_tl_status") or "na"
    if tl in ("skipped", "failed", "disabled"):
        bits.append("scene-fusion backup cohort — keep claims tied to vision/STT evidence.")
    return " ".join(bits)


def soft_boost_hero_class(
    hero_facts: Sequence[Dict[str, Any]],
    preferred_class: str,
) -> List[Dict[str, Any]]:
    """Stable soft re-order: preferred class first, original relative order preserved."""
    pref = str(preferred_class or "").strip().lower()
    facts = [f for f in hero_facts if isinstance(f, dict)]
    if not pref or not facts:
        return list(facts)
    head = [f for f in facts if str(f.get("class") or "").strip().lower() == pref]
    tail = [f for f in facts if str(f.get("class") or "").strip().lower() != pref]
    return head + tail


def apply_soft_hero_rank(
    ranked_facts: Sequence[Dict[str, Any]],
    ctx: Any,
) -> List[Dict[str, Any]]:
    """Apply soft class boost when flag on; identity fallback if pack absent."""
    facts = [f for f in ranked_facts if isinstance(f, dict)]
    arts = getattr(ctx, "output_artifacts", None) or {}
    flags, source = resolve_soft_flags(arts)
    pref = str(flags.get("pack_hero_class") or "")
    if not soft_bias_enabled() or not pref:
        if pref:
            logger.debug(
                "av_read soft bias shadow: would boost hero_class=%s source=%s (flag off)",
                pref,
                source,
            )
        return facts
    boosted = soft_boost_hero_class(facts, pref)
    if isinstance(arts, dict):
        arts["av_read_soft_bias_v1"] = {
            "hero_class": pref,
            "needs_deep": bool(flags.get("pack_needs_deep_teacher")),
            "applied": True,
            "source": source,
        }
    return boosted


def style_persona_recommend_from_flags(
    flags: Dict[str, Any],
    *,
    top_caption_style: Optional[str] = None,
    top_persona_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    P5 advisory recommendation only — never a silent pref write.
    Requires apply-optimized confirm=true to hard-apply.
    """
    out: Dict[str, Any] = {
        "status": "recommend_only",
        "requires_confirm": True,
        "silent_write": False,
        "items": [],
    }
    items: List[Dict[str, Any]] = []
    if top_caption_style:
        items.append(
            {
                "kind": "caption_style",
                "value": str(top_caption_style)[:48],
                "reason": (
                    "pack×outcome style winner (confirm to apply)"
                    if flags.get("pack_present")
                    else "attribution factor winner (confirm to apply; not pack-backed)"
                ),
            }
        )
    if top_persona_id:
        items.append(
            {
                "kind": "thumbnail_persona",
                "value": str(top_persona_id)[:64],
                "reason": (
                    "pack×studio persona winner (confirm to apply; no auto-write)"
                    if flags.get("pack_present")
                    else "studio/attribution persona hint (confirm to apply; no auto-write)"
                ),
            }
        )
    if flags.get("pack_needs_deep_teacher"):
        items.append(
            {
                "kind": "evidence_density",
                "value": "keep_deep_teacher",
                "reason": "fusion/deep-teacher cohort — keep TL/fusion on for similar clips",
            }
        )
    # Without pack_present, only emit items if we have style/persona hints (insights path).
    if not flags.get("pack_present") and not items:
        return out
    if not flags.get("pack_present") and not (top_caption_style or top_persona_id):
        return out
    out["items"] = items
    return out


__all__ = [
    "soft_bias_enabled",
    "coarse_pack_flags",
    "identity_soft_flags",
    "resolve_soft_flags",
    "soft_strategy_block",
    "soft_boost_hero_class",
    "apply_soft_hero_rank",
    "style_persona_recommend_from_flags",
]
