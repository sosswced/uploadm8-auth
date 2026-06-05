"""
Marketing plan generation: merges live metrics with either OpenAI JSON output
or deterministic templates (same response shape for admin UI).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

import httpx

from services.marketing_compliance import sanitize_truth_bundle_for_llm
from services.marketing_strategist_presets import (
    build_strategist_system_prompt,
    deterministic_copy_variants,
)

logger = logging.getLogger("uploadm8.marketing_strategist")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MARKETING_MODEL", "gpt-4o-mini")


def _deterministic_plan(payload: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    rk = str(payload.get("range") or payload.get("range_key") or "30d")
    kpis = metrics.get("kpis") or {}
    seg = metrics.get("segment_signals") or {}
    top = (metrics.get("ml_truth") or {}).get("top_strategies") or []
    et = metrics.get("engagement_truth") or {}
    eng_win = et.get("window") or {}
    er_pct = float(eng_win.get("avg_engagement_rate_pct") or 0)
    copy = deterministic_copy_variants(payload, metrics)

    conf = 58.0
    if kpis.get("nudge_ctr_pct", 0) >= 10:
        conf += 8
    if seg.get("token_pressure_accounts", 0) > 15:
        conf += 5
    if kpis.get("total_uploads", 0) > 100:
        conf += 4
    if int(eng_win.get("sample_uploads") or 0) > 80 and er_pct > 0:
        conf += 3
    conf = min(94.0, conf)

    platform_cov = 0.0
    pk = metrics.get("platform_kpis") or []
    if pk and kpis.get("total_uploads"):
        platform_cov = min(100.0, 100.0 * len(pk) / 4.0)

    execution = list(copy.get("execution_plan") or [])
    execution.append(
        "Rank platforms by upload engagement_rate_pct from engagement_truth.platform_upload_engagement; "
        "push growth budget to top surfaces."
    )
    execution.extend([f"ML prior: {s}" for s in top[:5]])

    return {
        "game_plan": {
            "north_star": {
                "platform_views": int(kpis.get("active_users", 0) * 120),
                "platform_engagement_rate_pct": round(
                    min(20.0, max(float(er_pct), 3.0 + float(kpis.get("nudge_ctr_pct", 0) or 0) * 0.12)),
                    2,
                ),
            },
            "data_quality": {"platform_coverage_pct": round(platform_cov, 2)},
            "confidence_score": round(conf, 1),
        },
        "newsletter": {"subject_lines": copy.get("subjects") or []},
        "offers": copy.get("offers") or [],
        "execution_plan": execution,
        "suggested_campaign": copy.get("suggested_campaign") or {},
    }


async def generate_marketing_plan(payload: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """
    Returns (plan, used_openai, raw_for_debug).
    """
    base = _deterministic_plan(payload, metrics)
    if not (OPENAI_API_KEY or "").strip():
        return base, False, {}

    allow_pii = bool(
        payload.get("allow_pii_in_llm") or payload.get("allow_pii_in_ml") or payload.get("allowPiiInMl")
    )
    metrics_for_llm = metrics if allow_pii else sanitize_truth_bundle_for_llm(metrics)

    try:
        system = build_strategist_system_prompt(payload)
        user_content = json.dumps({"request": payload, "metrics": metrics_for_llm}, default=str)[:12000]
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": OPENAI_MODEL,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.35,
                },
            )
        if r.status_code >= 400:
            logger.warning("OpenAI marketing strategist HTTP %s", r.status_code)
            return base, False, {"error": r.text[:400]}
        data = r.json()
        txt = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            return base, False, {}
        merged = _deterministic_plan(payload, metrics)
        for k in ("game_plan", "newsletter", "offers", "execution_plan"):
            if k in parsed and parsed[k]:
                merged[k] = parsed[k]
        # Merge suggested_campaign field-by-field so a partial model reply does not drop
        # channel, tiers, or numeric gates (those would break nudge targeting + deploy).
        parsed_sc = parsed.get("suggested_campaign")
        if isinstance(parsed_sc, dict) and parsed_sc:
            base_sc = dict(merged.get("suggested_campaign") or {})
            for sk, sv in parsed_sc.items():
                if sv is None:
                    continue
                if isinstance(sv, (list, dict)) and len(sv) == 0:
                    continue
                base_sc[sk] = sv
            merged["suggested_campaign"] = base_sc
        return merged, True, {"model": OPENAI_MODEL}
    except Exception as e:
        logger.warning("OpenAI marketing strategist failed: %s", e)
        return base, False, {"error": str(e)[:200]}
