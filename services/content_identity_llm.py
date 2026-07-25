"""LLM resolution layer for ``content_identity_v1``.

One small structured call per upload that names the footage subject in open
vocabulary and ranks hero facts — the universal handler for content the
deterministic layer (and any fixed taxonomy) has never seen.

Contract:
  * Input is ONLY harvested provider evidence — the model may not add facts.
  * Output is validated by ``core.content_identity.merge_llm_identity``:
    ungrounded facts are dropped, speed claims are consensus-gated.
  * Fail-soft: any error returns ``None`` and the upload continues on the
    deterministic identity. This call must never block or fail a pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("uploadm8-content-identity")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
IDENTITY_MODEL = os.environ.get("OPENAI_IDENTITY_MODEL", "gpt-4o-mini")
IDENTITY_MAX_TOKENS = 500
IDENTITY_TIMEOUT_SEC = float(os.environ.get("CONTENT_IDENTITY_TIMEOUT_SEC", "10"))


def _evidence_block(evidence: Dict[str, Any]) -> str:
    """Compact provider-grouped token listing for the prompt."""
    by_provider: Dict[str, List[str]] = {}
    for tok in (evidence.get("tokens") or [])[:60]:
        provider = str(tok.get("provider") or "unknown")
        kind = str(tok.get("kind") or "")
        text = str(tok.get("text") or "").strip()
        if text:
            by_provider.setdefault(provider, []).append(f"[{kind}] {text}")
    lines: List[str] = []
    for provider, items in by_provider.items():
        lines.append(f"{provider}:")
        lines.extend(f"  - {item}" for item in items[:16])
    for provider, text in (evidence.get("prose") or {}).items():
        snippet = str(text or "").strip()[:400]
        if snippet:
            lines.append(f"{provider} (prose): {snippet}")
    return "\n".join(lines)


def _speed_contract(speed_consensus: Optional[Dict[str, Any]]) -> str:
    sc = speed_consensus or {}
    try:
        peak = float(sc.get("peak_mph") or 0)
    except (TypeError, ValueError):
        peak = 0.0
    conf = str(sc.get("confidence") or "none")
    if peak >= 10 and conf == "high":
        return (
            f"SPEED CONTRACT: the only publishable speed is {peak:.0f} MPH "
            "(verified). Any other speed number is forbidden."
        )
    return (
        "SPEED CONTRACT: there is NO verified speed for this footage. "
        "Never state any speed number, even if a provider token mentions one."
    )


def build_identity_prompt(
    evidence: Dict[str, Any],
    speed_consensus: Optional[Dict[str, Any]],
) -> str:
    return f"""You are a video content identifier. Below is machine-harvested evidence about ONE uploaded video, grouped by provider (twelvelabs/fusion = scene understanding prose, vision = Google Vision, video_intelligence = Google VI tracks, audio = music ID + sound classes, speech = transcript, telemetry/osd = GPS/HUD).

EVIDENCE (the ONLY permitted source of facts — every word of your output must trace to a token below):
{_evidence_block(evidence)}

{_speed_contract(speed_consensus)}

Identify what this footage actually is, in open vocabulary — do NOT force it into a preset category. If it is something unusual, describe it precisely.

Return STRICT JSON with exactly these keys:
{{
  "subject": "concrete noun phrase naming what the footage shows (max 12 words)",
  "activity": "what is happening (max 10 words, empty string if unclear)",
  "setting": "where / environment (max 8 words, empty string if unclear)",
  "domain_tags": [{{"tag": "lowercase content domain, e.g. gardening / automotive / glassblowing", "confidence": 0.0}}],
  "hero_facts": [{{"text": "specific thumbnail/caption-worthy fact from evidence (max 10 words)", "class": "entity|landmark|logo|place|music|on_screen_text|count|transcript|speed|sound", "providers": ["which providers support it"]}}],
  "peak_metric_candidates": ["short numeric/superlative facts from evidence"],
  "do_not_invent": ["things NOT present that generation must not add"],
  "novel_content": false
}}

Rules:
- 1-3 domain_tags, confidence honest (0-1). Set novel_content=true when the content fits no common domain.
- 2-6 hero_facts ranked most compelling first. Facts must be VERIFIABLE from evidence tokens — no invented numbers, names, or drama.
- Never merge unrelated tokens into one fact."""


def parse_identity_response(raw: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("subject") is not None else None


async def resolve_content_identity(
    evidence: Dict[str, Any],
    speed_consensus: Optional[Dict[str, Any]] = None,
    *,
    upload_id: str = "",
    model: str = "",
    timeout_sec: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """One structured OpenAI call resolving open-vocabulary content identity.

    Returns the raw (unvalidated) dict, or ``None`` on any failure. Callers
    must merge through ``core.content_identity.merge_llm_identity`` — never
    trust this output directly.
    """
    if not OPENAI_API_KEY:
        logger.debug("content identity: OPENAI_API_KEY unset — deterministic only")
        return None
    if not (evidence.get("tokens") or evidence.get("prose")):
        return None

    prompt = build_identity_prompt(evidence, speed_consensus)
    payload = {
        "model": model or IDENTITY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": IDENTITY_MAX_TOKENS,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    timeout = timeout_sec or IDENTITY_TIMEOUT_SEC

    try:
        from stages.outbound_rl import outbound_slot

        async with outbound_slot("openai"):
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        if resp.status_code != 200:
            body = (resp.text or "")[:300]
            if resp.status_code == 429 or "insufficient_quota" in body.lower():
                logger.warning("content identity HTTP %s (quota): %s", resp.status_code, body)
            else:
                logger.warning("content identity HTTP %s: %s", resp.status_code, body)
            return None
        content = (
            (resp.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
        out = parse_identity_response(content)
        if out is None:
            logger.warning("content identity: unparseable JSON for upload=%s", upload_id)
        return out
    except Exception as e:  # fail-soft by contract — the pipeline never blocks on identity
        logger.warning("content identity call failed (non-fatal) upload=%s: %s", upload_id, e)
        return None


__all__ = [
    "resolve_content_identity",
    "build_identity_prompt",
    "parse_identity_response",
]
