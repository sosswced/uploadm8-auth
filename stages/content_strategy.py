from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import JobContext
from .safe_parse import json_dict

logger = logging.getLogger("uploadm8-worker.content_strategy")

_POLICY_PATH = Path(__file__).with_name("content_style_policy.json")


def _load_policy() -> Dict[str, Any]:
    try:
        raw = _POLICY_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("content_strategy: could not read %s: %s", _POLICY_PATH.name, e)
        return {"defaults": {}, "rules": []}
    obj = json_dict(raw, default={"defaults": {}, "rules": []}, context="content_style_policy.json")
    obj.setdefault("defaults", {})
    obj.setdefault("rules", [])
    return obj


def _safe_slug(s: Any) -> str:
    return str(s or "").strip().lower().replace("-", "_")


def _merge_constraints(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        out[k] = v
    return out


def _score_match(rule_match: Dict[str, Any], category: str, emotion: str, platform: str, audience_intent: str) -> int:
    score = 0
    pairs = [
        ("category", category),
        ("emotion", emotion),
        ("platform", platform),
        ("audience_intent", audience_intent),
    ]
    for key, value in pairs:
        expected = _safe_slug(rule_match.get(key))
        if not expected or expected in ("*", "any"):
            continue
        if expected != _safe_slug(value):
            return -1
        score += 1
    return score


def _derive_scene_tags(ctx: JobContext) -> List[str]:
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vi = getattr(ctx, "video_intelligence_context", None) or {}
    tags: List[str] = []
    tags.extend([str(x).strip().lower() for x in (ac.get("suggested_keywords") or []) if str(x).strip()])
    tags.extend([str(x).strip().lower() for x in (vc.get("label_names") or [])[:20] if str(x).strip()])
    for x in (vc.get("landmark_names") or [])[:6]:
        s = str(x).strip().lower()
        if s:
            tags.append(s)
    for x in (vc.get("logo_names") or [])[:6]:
        s = str(x).strip().lower()
        if s:
            tags.append(s)
    if isinstance(vi, dict) and not vi.get("error"):
        for row in (vi.get("segment_labels") or [])[:8]:
            if isinstance(row, dict):
                d = str(row.get("description") or "").strip().lower()
                if d:
                    tags.append(d)
    tags.extend([str(x).strip().lower() for x in (ac.get("yamnet_events") or [])[:10] if str(x).strip()])
    # deterministic dedupe
    out: List[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:40]


def _derive_motion_intensity(ctx: JobContext) -> str:
    tel = ctx.telemetry or ctx.telemetry_data
    mph = float(getattr(tel, "max_speed_mph", 0.0) or 0.0) if tel else 0.0
    if mph >= 70:
        return "high"
    if mph >= 25:
        return "medium"
    return "low"


def _derive_pace(ctx: JobContext) -> str:
    ac = ctx.audio_context or {}
    role = _safe_slug(ac.get("transcript_role"))
    if role in ("third_party_lyrics", "mixed_speech_and_music"):
        return "fast"
    if len((ac.get("transcript") or "").strip()) > 280:
        return "medium"
    return "slow"


def map_ui_caption_style_for_strategy(style: str) -> str:
    """
    Map settings-page captionStyle (story | punchy | factual) to policy / M8 style slugs.
    Policy JSON uses e.g. story, hook, educational; unknown values fall back to story.
    """
    s = _safe_slug(style)
    if s == "punchy":
        return "hook"
    if s == "factual":
        return "factual"
    if s in ("story", "hook", "educational"):
        return s
    return "story"


def map_ui_caption_tone_for_strategy(tone: str) -> str:
    """
    Map settings-page captionTone (hype | calm | cinematic | authentic) to policy slugs.
    Rules in content_style_policy.json use e.g. bold, professional, authentic.
    """
    t = _safe_slug(tone)
    if t == "hype":
        return "bold"
    if t in ("calm", "cinematic", "authentic"):
        return t
    return "authentic"


def map_ui_caption_voice_to_persona(voice: str) -> str:
    """
    Map settings-page captionVoice to M8 _persona_system_prompt keys:
    storyteller | creator_coach | hype_friend | expert_analyst
    """
    v = _safe_slug(voice).replace("-", "_")
    return {
        "default": "storyteller",
        "mentor": "creator_coach",
        "hypebeast": "hype_friend",
        "best_friend": "storyteller",
        "teacher": "creator_coach",
        "cinematic_narrator": "storyteller",
    }.get(v, "storyteller")


def build_content_strategy(
    ctx: JobContext,
    *,
    category: str,
    audience_intent: str = "",
    platforms: List[str] | None = None,
    user_caption_style: Optional[str] = None,
    user_caption_tone: Optional[str] = None,
    user_caption_voice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build ML/policy strategy for M8 caption prompts.

    Product rule: **Account caption settings seed the strategy master** (mapped into policy
    slugs). Rows in ``content_style_policy.json`` may still **override** style/tone/persona
    per platform when their match keys (category, emotion, platform, audience_intent) fit.
    ML priors attached elsewhere remain soft bias only.
    """
    policy = _load_policy()
    defaults = dict(policy.get("defaults") or {})
    rules = list(policy.get("rules") or [])
    ac = ctx.audio_context or {}

    emotion = _safe_slug(ac.get("emotional_tone") or ac.get("hume_emotions", {}).get("dominant_emotion") or "neutral")
    audience = _safe_slug(audience_intent or defaults.get("audience_intent") or "engagement")
    scene_tags = _derive_scene_tags(ctx)
    motion_intensity = _derive_motion_intensity(ctx)
    pace = _derive_pace(ctx)

    target_platforms = [str(p).lower() for p in (platforms or ctx.platforms or ["tiktok"])]
    if not target_platforms:
        target_platforms = ["tiktok"]

    base_constraints = dict((defaults.get("constraints") or {}))
    user_seeded = any(
        x is not None for x in (user_caption_style, user_caption_tone, user_caption_voice)
    )
    if user_seeded:
        master = {
            "caption_style": map_ui_caption_style_for_strategy(user_caption_style or "story"),
            "tone": map_ui_caption_tone_for_strategy(user_caption_tone or "authentic"),
            "persona": map_ui_caption_voice_to_persona(user_caption_voice or "default"),
            "risk_level": _safe_slug(defaults.get("risk_level") or "safe"),
        }
    else:
        master = {
            "caption_style": _safe_slug(defaults.get("caption_style") or "story"),
            "tone": _safe_slug(defaults.get("tone") or "authentic"),
            "persona": _safe_slug(defaults.get("persona") or "storyteller"),
            "risk_level": _safe_slug(defaults.get("risk_level") or "safe"),
        }

    platform_targets: Dict[str, Any] = {}
    for pl in target_platforms:
        selected = {
            "caption_style": master["caption_style"],
            "tone": master["tone"],
            "persona": master["persona"],
            "risk_level": master["risk_level"],
            "audience_intent": audience,
            "constraints": dict(base_constraints),
        }
        best_score = -1
        for rule in rules:
            m = (rule or {}).get("match") or {}
            sc = _score_match(m, category=category, emotion=emotion, platform=pl, audience_intent=audience)
            if sc < 0 or sc < best_score:
                continue
            out = (rule or {}).get("output") or {}
            merged = dict(selected)
            for key in ("caption_style", "tone", "persona", "risk_level", "audience_intent"):
                if out.get(key):
                    merged[key] = _safe_slug(out.get(key))
            merged["constraints"] = _merge_constraints(selected.get("constraints") or {}, out.get("constraints") or {})
            selected = merged
            best_score = sc
        platform_targets[pl] = selected

    # Global strategy keeps deterministic "master intent" anchored to first platform.
    anchor = platform_targets.get(target_platforms[0], {})
    inputs: Dict[str, Any] = {
        "category": category,
        "scene_tags": scene_tags,
        "pace": pace,
        "motion_intensity": motion_intensity,
        "sentiment_emotion": emotion,
        "transcript_highlights": (ac.get("transcript") or "")[:240],
        "audience_intent": audience,
        "platforms": target_platforms,
    }
    if user_seeded:
        inputs["user_caption_seed"] = {
            "style_ui": user_caption_style,
            "tone_ui": user_caption_tone,
            "voice_ui": user_caption_voice,
            "mapped_master": {
                "caption_style": master["caption_style"],
                "tone": master["tone"],
                "persona": master["persona"],
            },
        }
    strategy = {
        "version": "1.1",
        "inputs": inputs,
        "outputs": {
            "caption_style": anchor.get("caption_style", master["caption_style"]),
            "tone": anchor.get("tone", master["tone"]),
            "voice_persona": anchor.get("persona", master["persona"]),
            "risk_level": anchor.get("risk_level", master["risk_level"]),
            "platform_targets": platform_targets,
        },
    }
    return strategy
