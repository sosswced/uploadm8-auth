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

logger = logging.getLogger("uploadm8.marketing_strategist")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MARKETING_MODEL", "gpt-4o-mini")


def _deterministic_plan(payload: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    rk = str(payload.get("range") or payload.get("range_key") or "30d")
    objective = str(payload.get("objective") or "revenue_growth")
    kpis = metrics.get("kpis") or {}
    seg = metrics.get("segment_signals") or {}
    top = (metrics.get("ml_truth") or {}).get("top_strategies") or []
    et = metrics.get("engagement_truth") or {}
    eng_win = et.get("window") or {}
    er_pct = float(eng_win.get("avg_engagement_rate_pct") or 0)

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
        "newsletter": {
            "subject_lines": [
                f"Your upload funnel — {rk} pulse",
                "More reach: thumbnails + multi-platform",
                f"Ops note: {seg.get('token_pressure_accounts', 0)} accounts near token floor",
            ]
        },
        "offers": [
            {"name": "PUT + AIC refill bundle", "value_prop": "Micro-transaction for heavy batch weeks"},
            {"name": "Creator Lite trial nudge", "value_prop": f"Target {seg.get('free_high_intent_uploaders', 0)} active free uploaders"},
            {"name": "Studio expansion", "value_prop": f"{seg.get('expansion_ready_accounts', 0)} multi-platform accounts"},
        ],
        "execution_plan": [
            "Prioritize in-app nudges for low-PUT cohort while CTR holds.",
            "Email cohort: clicked a nudge but no revenue in 7d.",
            "Discord: spotlight Thumbnail Studio workflows using live Pikzels usage data.",
            "Surface 'views vs cohort' coaching on dashboard for underperforming uploaders.",
            f"Engagement truth ({rk}): avg (likes+comments+shares)/views ~{er_pct:.2f}% on {int(eng_win.get('sample_uploads') or 0)} uploads — double down where reactions beat baseline.",
            "Rank platforms by upload engagement_rate_pct from engagement_truth.platform_upload_engagement; push growth budget to top surfaces.",
        ]
        + [f"ML prior: {s}" for s in top[:5]],
        "suggested_campaign": {
            "name": "Auto: data-backed revenue push",
            "objective": objective,
            "channel": str(payload.get("channel_mix") or "mixed"),
            "range": rk,
            "min_uploads_30d": 4 if seg.get("free_high_intent_uploaders", 0) > 6 else 2,
            "min_enterprise_fit_score": 45,
            "min_nudge_ctr_pct": max(5.0, min(12.0, float(kpis.get("nudge_ctr_pct", 8) or 8) * 0.6)),
            "tiers": ["free", "creator_lite"] if objective != "enterprise_expansion" else ["creator_pro", "studio"],
            "require_no_revenue_7d": True,
            "notes": (
                "Generated from marketing_events, revenue_tracking, uploads, studio_usage_events, "
                "and engagement_truth (views/likes/comments/shares)."
            ),
        },
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
        system = (
            "You are a B2C SaaS growth strategist. Given JSON metrics, output a single JSON object "
            "with keys: game_plan (north_star, data_quality, confidence_score), newsletter (subject_lines array), "
            "offers (array of {name, value_prop}), execution_plan (string array), suggested_campaign "
            "(name, objective, channel, range, min_uploads_30d int, min_enterprise_fit_score int, "
            "min_nudge_ctr_pct float, tiers string array, require_no_revenue_7d bool, notes string). "
            "Keep arrays concise; confidence_score 0-100 based on signal strength. "
            "Do not invent user emails or PII; metrics may be aggregated only."
        )
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
        for k in ("game_plan", "newsletter", "offers", "execution_plan", "suggested_campaign"):
            if k in parsed and parsed[k]:
                merged[k] = parsed[k]
        return merged, True, {"model": OPENAI_MODEL}
    except Exception as e:
        logger.warning("OpenAI marketing strategist failed: %s", e)
        return base, False, {"error": str(e)[:200]}
