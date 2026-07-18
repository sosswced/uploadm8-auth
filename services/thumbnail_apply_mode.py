"""Thumbnail Studio → Upload apply modes, XOR rules, and strategy binding.

Apply modes (upload-time):
  - fresh_generate: frame + strategy; YT support + persona subject to ref_persona_mode
  - strategy_only: frame + strategy; never attach YouTube support image
  - pinned_cover: use Studio variant R2 image for YouTube 16:9; adapt 9:16 covers

Ref/persona modes (product copy enforcement):
  - recreate_style: support image, no persona
  - face_brand: persona, no support image
  - both: allow both (warn via diagnostics)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

# Public UX names + bridge aliases (cover_direct / support_image from Studio feedback).
APPLY_MODES = ("fresh_generate", "strategy_only", "pinned_cover")
REF_PERSONA_MODES = ("recreate_style", "face_brand", "both")

_DEFAULT_APPLY = "fresh_generate"
_DEFAULT_REF_PERSONA = "recreate_style"


def normalize_apply_mode(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "fresh": "fresh_generate",
        "generate": "fresh_generate",
        "support_image": "fresh_generate",
        "strategy": "strategy_only",
        "pinned": "pinned_cover",
        "pin": "pinned_cover",
        "cover": "pinned_cover",
        "cover_direct": "pinned_cover",
    }
    s = aliases.get(s, s)
    return s if s in APPLY_MODES else _DEFAULT_APPLY


def to_bridge_apply_mode(raw: Any) -> str:
    """Map UX apply mode → ``thumbnail_studio_upload_bridge`` mode strings."""
    mode = normalize_apply_mode(raw)
    if mode == "pinned_cover":
        return "cover_direct"
    if mode == "strategy_only":
        return "strategy_only"
    return "support_image"


def normalize_ref_persona_mode(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "recreate": "recreate_style",
        "youtube": "recreate_style",
        "support": "recreate_style",
        "persona": "face_brand",
        "face": "face_brand",
        "brand": "face_brand",
        "combined": "both",
        "all": "both",
    }
    s = aliases.get(s, s)
    return s if s in REF_PERSONA_MODES else _DEFAULT_REF_PERSONA


def resolve_ref_persona_mode(
    us: Optional[Mapping[str, Any]],
    *,
    apply_mode: Optional[str] = None,
) -> str:
    """Derive ref/persona XOR mode from prefs + apply mode."""
    prefs = us if isinstance(us, Mapping) else {}
    mode = normalize_apply_mode(
        apply_mode
        or prefs.get("thumbnail_apply_mode")
        or prefs.get("thumbnailApplyMode")
    )
    explicit = prefs.get("thumbnail_ref_persona_mode") or prefs.get("thumbnailRefPersonaMode")
    if explicit:
        return normalize_ref_persona_mode(explicit)
    if mode == "strategy_only":
        # No YT support; persona optional → face_brand semantics for XOR
        return "face_brand"
    if mode == "pinned_cover":
        # Pinned asset is the look; avoid mixing persona into YT pin
        return "recreate_style"
    # fresh_generate: if persona enabled prefer face_brand unless user chose both
    persona_on = bool(
        prefs.get("thumbnail_persona_enabled")
        if "thumbnail_persona_enabled" in prefs
        else prefs.get("thumbnailPersonaEnabled")
    )
    if persona_on:
        return "face_brand"
    return "recreate_style"


def allow_youtube_support_image(us: Optional[Mapping[str, Any]], *, apply_mode: Optional[str] = None) -> bool:
    mode = normalize_apply_mode(
        apply_mode
        or (us or {}).get("thumbnail_apply_mode")
        or (us or {}).get("thumbnailApplyMode")
    )
    if mode == "strategy_only":
        return False
    if mode == "pinned_cover":
        return False  # pin replaces support-image recreate path for YT
    rpm = resolve_ref_persona_mode(us, apply_mode=mode)
    return rpm in ("recreate_style", "both")


def allow_persona_on_render(us: Optional[Mapping[str, Any]], *, apply_mode: Optional[str] = None) -> bool:
    mode = normalize_apply_mode(
        apply_mode
        or (us or {}).get("thumbnail_apply_mode")
        or (us or {}).get("thumbnailApplyMode")
    )
    if mode == "pinned_cover":
        # Persona only when regenerating vertical covers (handled by caller);
        # YT pin itself never FaceSwaps.
        rpm = resolve_ref_persona_mode(us, apply_mode=mode)
        return rpm in ("face_brand", "both")
    rpm = resolve_ref_persona_mode(us, apply_mode=mode)
    return rpm in ("face_brand", "both")


def strategy_source_ids(strategy: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not isinstance(strategy, Mapping):
        return "", ""
    job = str(strategy.get("job_id") or strategy.get("source_job_id") or "").strip()
    var = str(strategy.get("variant_id") or strategy.get("source_variant_id") or "").strip()
    return job, var


def bind_source_ids_into_prefs(
    prefs: Dict[str, Any],
    *,
    job_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    strategy: Optional[Mapping[str, Any]] = None,
) -> None:
    """Ensure source_job_id / source_variant_id live on prefs and nested strategy."""
    j = str(job_id or "").strip()
    v = str(variant_id or "").strip()
    if not j or not v:
        sj, sv = strategy_source_ids(strategy if strategy is not None else prefs.get("thumbnail_studio_default_strategy"))
        j = j or sj
        v = v or sv
    if j:
        prefs["thumbnail_source_job_id"] = j
        prefs["thumbnailSourceJobId"] = j
    if v:
        prefs["thumbnail_source_variant_id"] = v
        prefs["thumbnailSourceVariantId"] = v
    strat = prefs.get("thumbnail_studio_default_strategy")
    if not isinstance(strat, dict):
        strat = prefs.get("thumbnailStudioDefaultStrategy")
    if isinstance(strat, dict) and (j or v):
        merged = dict(strat)
        if j:
            merged["job_id"] = j
            merged["source_job_id"] = j
        if v:
            merged["variant_id"] = v
            merged["source_variant_id"] = v
        prefs["thumbnail_studio_default_strategy"] = merged
        prefs["thumbnailStudioDefaultStrategy"] = merged


def strategy_summary_for_ui(strategy: Optional[Mapping[str, Any]], prefs: Optional[Mapping[str, Any]] = None) -> str:
    """Human line for upload page: Saved run · date · Variant N."""
    strat = dict(strategy) if isinstance(strategy, Mapping) else {}
    p = prefs if isinstance(prefs, Mapping) else {}
    job = str(
        p.get("thumbnail_source_job_id")
        or p.get("thumbnailSourceJobId")
        or strat.get("source_job_id")
        or strat.get("job_id")
        or ""
    ).strip()
    var = str(
        p.get("thumbnail_source_variant_id")
        or p.get("thumbnailSourceVariantId")
        or strat.get("source_variant_id")
        or strat.get("variant_id")
        or ""
    ).strip()
    parts = []
    layout = strat.get("layout_name") or strat.get("layout_pattern") or strat.get("format_key")
    if layout:
        parts.append(str(layout).replace("_", " "))
    niche = strat.get("audience_niche") or strat.get("niche")
    if niche:
        parts.append(f"niche {niche}")
    created = strat.get("created_at") or strat.get("selected_at")
    date_bit = ""
    if created:
        try:
            if isinstance(created, (int, float)):
                dt = datetime.fromtimestamp(float(created), tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            date_bit = dt.strftime("%b %d")
        except Exception:
            date_bit = ""
    if job or var:
        label = "Thumbnail strategy from Saved run"
        if date_bit:
            label += f" · {date_bit}"
        if var:
            short = var if len(var) <= 8 else var[:8]
            label += f" · Variant {short}"
        if parts:
            label += " (" + " · ".join(parts) + ")"
        return label
    if parts:
        return "Using Studio default: " + " · ".join(parts)
    return ""


def structured_strategy_payload(strategy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Fields to pass into Pikzels options / brief (not only prose notes)."""
    if not isinstance(strategy, Mapping) or not strategy:
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "format_key",
        "layout_name",
        "layout_pattern",
        "audience_niche",
        "reference_strength",
        "closeness",
        "text_position",
        "contrast_profile",
        "emotion",
        "face_scale",
        "hydration_focus",
        "competitor_gap_mode",
        "selected_headline_style",
    ):
        val = strategy.get(key)
        if val not in (None, "", []):
            out[key] = val
    # Normalize closeness → reference_strength
    if "reference_strength" not in out and "closeness" in out:
        try:
            out["reference_strength"] = int(out["closeness"])
        except (TypeError, ValueError):
            pass
    platforms = strategy.get("platforms")
    if isinstance(platforms, dict) and platforms:
        out["platforms"] = platforms
    return out


def platform_strategy_overlay(
    strategy: Optional[Mapping[str, Any]],
    platform: str,
) -> Dict[str, Any]:
    """Merge top-level strategy with optional per-platform map entry."""
    base = structured_strategy_payload(strategy)
    if not isinstance(strategy, Mapping):
        return base
    plats = strategy.get("platforms")
    if not isinstance(plats, dict):
        return base
    plat = str(platform or "").strip().lower()
    overlay = plats.get(plat) or plats.get(plat.replace("instagram", "ig"))
    if isinstance(overlay, Mapping) and overlay:
        merged = dict(base)
        merged.update(structured_strategy_payload(overlay))
        return merged
    return base


def apply_structured_strategy_to_brief(
    brief: Dict[str, Any],
    strategy: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Attach structured strategy fields onto the thumbnail brief for the renderer."""
    b = dict(brief or {})
    structured = structured_strategy_payload(strategy)
    if not structured:
        return b
    b["default_strategy"] = dict(strategy) if isinstance(strategy, Mapping) else {}
    b["_uploadm8_strategy_structured"] = structured
    if structured.get("format_key"):
        b["format_key"] = structured["format_key"]
    if structured.get("layout_pattern"):
        b["layout_pattern"] = structured["layout_pattern"]
    if structured.get("layout_name"):
        b["layout_name"] = structured["layout_name"]
    rs = structured.get("reference_strength")
    if isinstance(rs, (int, float)):
        b["_uploadm8_reference_strength"] = int(rs)
    return b


def merge_strategy_into_studio_options(
    options: Optional[Dict[str, Any]],
    strategy: Optional[Mapping[str, Any]],
    *,
    platform: str = "",
) -> Dict[str, Any]:
    opts = dict(options or {})
    structured = platform_strategy_overlay(strategy, platform) if platform else structured_strategy_payload(strategy)
    if not structured:
        return opts
    rs = structured.get("reference_strength")
    if isinstance(rs, (int, float)) and "reference_strength" not in opts:
        opts["reference_strength"] = int(rs)
    if structured.get("format_key"):
        opts["format_key"] = structured["format_key"]
    if structured.get("layout_pattern"):
        opts["layout_pattern"] = structured["layout_pattern"]
    if structured.get("layout_name"):
        opts["layout_name"] = structured["layout_name"]
    opts["strategy_structured"] = structured
    return opts


def estimate_pikzels_aic_hint(
    *,
    platform_count: int,
    engine_on: bool,
    apply_mode: str = "fresh_generate",
    has_persona: bool = False,
) -> Optional[str]:
    """Cheap client-facing estimate string (weights are approximate)."""
    if not engine_on or platform_count <= 0:
        return None
    mode = normalize_apply_mode(apply_mode)
    # Rough AIC units aligned with stages/ai_service_costs defaults (display only).
    per = 3
    if mode == "pinned_cover":
        # YT pin free of Pikzels; vertical platforms still may regenerate
        regen = max(0, platform_count - 1)
        aic = regen * per + (1 if has_persona and regen else 0)
    else:
        aic = platform_count * per
        if has_persona:
            aic += platform_count  # faceswap-ish
    if aic <= 0:
        return "Pinned YouTube cover — no Pikzels AIC for 16:9; vertical covers may still use AIC."
    return f"~{aic} AIC for Pikzels × {platform_count} platform{'s' if platform_count != 1 else ''} ({mode.replace('_', ' ')})"
