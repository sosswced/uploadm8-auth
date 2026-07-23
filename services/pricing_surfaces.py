"""
Canonical PUT/AIC debit surfaces for UI copy, wallet estimates, and sync scripts.

Single source of truth for *numbers* that marketing/guide/wallet hardcode.
Live billing still uses SERVICE_WEIGHTS + DB overrides at presign; this module
mirrors code defaults so ``scripts/sync_pricing_surfaces.py`` can regenerate
frontend artifacts after weight/pricing changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from services.billing_catalog import PUT_COST_DEFAULTS
from services.thumbnail_studio import estimate_pikzels_v2_call_cost, estimate_studio_cost
from stages.ai_service_costs import (
    SERVICE_WEIGHTS,
    compute_aic_service_charge,
    duration_multiplier,
    service_catalog,
)

CALIBRATION_ID = "2026-07-v2"

# Default-light upload (matches UNIVERSAL baseline: no Whisper / TL / VI / dashcam).
_LIGHT_SERVICES: Set[str] = {
    "caption_llm",
    "thumbnail_ai",
    "vision_google",
    "audio_gpt_classify",
    "audio_yamnet",
    "audio_acr",
    "trend_intel",
    "telemetry_trill",
}

_FULL_SERVICES: Set[str] = set(SERVICE_WEIGHTS.keys()) - {
    "thumbnail_recreate_ai",
    "persona_consistency",
    "thumbnail_ctr_ranker",
    "marketing_image",
    "thumbnail_competitor_gap",
}


def _aic_for(enabled: Set[str], *, duration_s: float = 60.0, frames: int = 5, thumbs: int = 1) -> int:
    return int(
        compute_aic_service_charge(
            enabled=enabled & set(SERVICE_WEIGHTS.keys()),
            duration_seconds=duration_s,
            max_caption_frames=frames,
            num_thumbnails=thumbs,
            weights=SERVICE_WEIGHTS,
        )
    )


def build_pricing_surfaces_snapshot() -> Dict[str, Any]:
    """Serializable snapshot used by sync script, /api/pricing, and tests."""
    put = {k: int(v) for k, v in PUT_COST_DEFAULTS.items()}
    light_aic = _aic_for(_LIGHT_SERVICES, duration_s=60.0, frames=5)
    full_aic = _aic_for(_FULL_SERVICES, duration_s=60.0, frames=8)
    studio1_put, studio1_aic, _ = estimate_studio_cost(
        variant_count=1,
        has_persona=False,
        competitor_gap_mode=False,
        has_channel_memory=False,
    )
    pikzels: Dict[str, Dict[str, int]] = {}
    for op in (
        "prompt",
        "recreate",
        "edit",
        "one_click_fix",
        "faceswap",
        "score",
        "titles",
        "persona",
        "style",
    ):
        p, a, _ = estimate_pikzels_v2_call_cost(op)
        pikzels[op] = {"put": int(p), "aic": int(a)}

    recreate_aic = int(pikzels.get("recreate", {}).get("aic") or 90)
    estimates = {
        "put_base": int(put.get("base", 10)),
        "put_typical": int(put.get("base", 10)),  # 1 platform, no priority
        "aic_light_60s": light_aic,
        "aic_full_60s": full_aic,
        "aic_light_lo": max(1, light_aic - 2),
        "aic_light_hi": light_aic + 1,
        "aic_full_lo": max(1, full_aic - 3),
        "aic_full_hi": full_aic + 1,
        "studio_recreate_1_put": int(studio1_put),
        "studio_recreate_1_aic": int(studio1_aic),
        "pikzels_recreate_aic": recreate_aic,
        "duration_multiplier_60s": round(float(duration_multiplier(60.0)), 4),
    }

    copy = _build_copy(put, estimates, pikzels)
    return {
        "schema_version": 1,
        "calibration": CALIBRATION_ID,
        "anchor": {
            "aic_retail_usd": 0.01,
            "note": "1 AIC ≈ $0.01 intended retail; Pikzels/Studio ≥ 2.5× vendor",
        },
        "put_cost_rules": put,
        "service_weights": {k: int(v) for k, v in SERVICE_WEIGHTS.items()},
        "aic_pipeline_catalog": service_catalog(SERVICE_WEIGHTS),
        "pikzels_v2": pikzels,
        "estimates": estimates,
        "copy": copy,
        "html_snippets": _html_snippets(put, estimates),
        "surfaces": [
            "frontend/js/pricing-surfaces.generated.js",
            "frontend/wallet-tokens.js (reads generated estimates)",
            "frontend/js/marketing-copy.js (FAQ from generated copy)",
            "frontend/guide.html (um8-pricing-sync markers)",
            "frontend/settings.html (um8-pricing-sync markers)",
            "GET /api/pricing → debit_surfaces",
        ],
    }


def _build_copy(put: Dict[str, int], est: Dict[str, Any], pikzels: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    base = int(put.get("base", 10))
    extra = int(put.get("per_extra_platform", 2))
    pri = int(put.get("priority_lane_addon", 5))
    thumb = int(put.get("per_extra_thumbnail_beyond_first", 1))
    lo, hi = est["aic_light_lo"], est["aic_light_hi"]
    flo, fhi = est["aic_full_lo"], est["aic_full_hi"]
    rec = int(est["pikzels_recreate_aic"])
    return {
        "faq_put_aic": (
            f"Publishing credits (PUT) pay for goes-live work — base {base} per job, "
            f"plus {extra} per extra destination, +{pri} priority, +{thumb} per extra thumbnail. "
            f"AI credits (AIC) pay for the AI services you enable (captions, Vision, optional "
            f"Twelve Labs / Video Intelligence, audio helpers) and Thumbnail Studio / Pikzels. "
            f"Speech-to-text (Whisper) is included at no extra AIC when enabled. "
            f"Defaults stay light (~{lo}–{hi} AIC per short); heavy analyzers are opt-in "
            f"(~{flo}–{fhi} AIC full-smart). Studio recreate is about {rec} AIC per image. "
            f"Balances only move when work runs. Subscription credits renew each cycle; "
            f"add-on packs never expire."
        ),
        "settings_put_blurb": (
            f"Consumed each time you process & post a video. Base = <strong>{base} PUT</strong>, "
            f"+{extra} per extra <strong>publish target</strong> (each platform when no groups, "
            f"or each selected account when using groups), +{pri} for priority lane, "
            f"+{thumb} per extra thumbnail.<br>"
            f"<em>Example: 3 publish targets → {base} + {2 * extra} = "
            f"<strong>{base + 2 * extra} PUT</strong></em>"
        ),
        "settings_aic_blurb": (
            f"Consumed by the AI services you enable (captions, thumbnails, Vision, optional "
            f"Twelve Labs / Video Intelligence, audio helpers). Speech-to-text (Whisper) is "
            f"included at no extra AIC. Each paid service has a weight; longer clips cost more "
            f"for minute-metered tools. Defaults keep heavy analyzers off so a typical short "
            f"stays near ~{lo}–{hi} AIC; turning everything on lands near ~{flo}–{fhi} AIC "
            f"for a 60s clip.<br>"
            f"<em>Example: captions + thumbnails + Vision (defaults) → about "
            f"<strong>{lo}–{hi} AIC</strong>. Thumbnail Studio / Pikzels recreate is priced "
            f"separately (~{rec} AIC per image, ~2.5× provider cost).</em>"
        ),
        "guide_aic_desc": (
            f"Used for the AI services you enable — captions, hashtags, Vision, optional "
            f"Twelve Labs / Video Intelligence, audio helpers, and Thumbnail Studio / Pikzels. "
            f"Speech-to-text (Whisper) is included at no extra AIC when enabled. Each paid "
            f"service has a weight; longer clips cost more for minute-metered tools. Defaults "
            f"keep heavy analyzers off (~{lo}–{hi} AIC for a typical short); full-smart is "
            f"~{flo}–{fhi} AIC. Studio recreate is priced separately (~{rec} AIC per image)."
        ),
        "whisper_note": (
            "Speech-to-text (Whisper) is off by default and included at no extra AIC when "
            "enabled — turn it on for talking-head and voiceover clips."
        ),
    }


def _html_snippets(put: Dict[str, int], est: Dict[str, Any]) -> Dict[str, str]:
    base = int(put.get("base", 10))
    extra = int(put.get("per_extra_platform", 2))
    pri = int(put.get("priority_lane_addon", 5))
    thumb = int(put.get("per_extra_thumbnail_beyond_first", 1))
    return {
        "guide_put_table_rows": (
            f"<tr><td>Base upload</td><td><strong>{base}</strong></td></tr>\n"
            f"<tr><td>Each extra destination after the first</td><td><strong>+{extra}</strong></td></tr>\n"
            f"<tr><td>Faster priority processing (paid plans)</td><td><strong>+{pri}</strong></td></tr>\n"
            f"<tr><td>Each extra thumbnail after the first</td><td><strong>+{thumb}</strong></td></tr>"
        ),
        "guide_aic_table_rows": (
            f"<tr><td>Defaults (captions + thumbs + Vision)</td>"
            f"<td><strong>~{est['aic_light_lo']}–{est['aic_light_hi']}</strong></td></tr>\n"
            f"<tr><td>Full-smart 60s (all services on)</td>"
            f"<td><strong>~{est['aic_full_lo']}–{est['aic_full_hi']}</strong></td></tr>\n"
            f"<tr><td>Thumbnail Studio / Pikzels recreate</td>"
            f"<td><strong>~{est['pikzels_recreate_aic']}</strong> per image</td></tr>"
        ),
    }


def render_generated_js(snapshot: Dict[str, Any] | None = None) -> str:
    """JavaScript module text for frontend/js/pricing-surfaces.generated.js."""
    import json

    snap = snapshot or build_pricing_surfaces_snapshot()
    payload = json.dumps(snap, indent=2, sort_keys=True)
    return (
        "/**\n"
        f" * AUTO-GENERATED by scripts/sync_pricing_surfaces.py — {snap.get('calibration')}\n"
        " * Do not edit by hand. Re-run: python scripts/sync_pricing_surfaces.py\n"
        " */\n"
        "(function (global) {\n"
        "  'use strict';\n"
        f"  var SNAPSHOT = {payload};\n"
        "  global.UploadM8PricingSurfaces = SNAPSHOT;\n"
        "  if (typeof module !== 'undefined' && module.exports) {\n"
        "    module.exports = SNAPSHOT;\n"
        "  }\n"
        "})(typeof window !== 'undefined' ? window : globalThis);\n"
    )
