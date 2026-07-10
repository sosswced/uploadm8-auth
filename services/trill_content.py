"""OpenAI title/caption/hashtag generation from Trill telemetry metrics only."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from core.config import OPENAI_API_KEY, TRILL_SYSTEM_PROMPT

logger = logging.getLogger("uploadm8-api")


def generate_trill_content(trill_metadata: dict, user_prefs: dict = None) -> dict:
    """
    Use OpenAI to generate titles, captions, and hashtags from **Trill telemetry analysis**.

    ``trill_metadata`` comes from ``telemetry_trill.safe_analyze_video`` (includes
    Census nearest place, PAD-US protected-lands hit (PostGIS), speed, curvature, etc.
    when configured). This is **not** the full upload worker pipeline
    (no Vision / audio stack / M8).

    Returns:
        {
            "title": "Generated title",
            "caption": "Generated caption",
            "hashtags": ["tag1", "tag2", ...],
            "tokens_used": 150,
            "model": "gpt-4o-mini"
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    import openai

    openai.api_key = OPENAI_API_KEY

    score = trill_metadata.get("trill_score", 0)
    bucket = trill_metadata.get("speed_bucket", "CRUISE MODE")
    place = trill_metadata.get("place_name", "")
    state = trill_metadata.get("state", "")
    protected_name = trill_metadata.get("protected_name")
    near_protected = trill_metadata.get("near_protected", False)
    elev_gain = trill_metadata.get("elev_gain_m", 0)
    curv_score = trill_metadata.get("curv_score", 0)
    dyn_score = trill_metadata.get("dyn_score", 0)
    turny = trill_metadata.get("turny", False)
    spirited = trill_metadata.get("spirited", False)
    state_usps = (trill_metadata.get("state_usps") or "").strip()
    max_mph = trill_metadata.get("max_speed_mph")
    avg_mph = trill_metadata.get("avg_speed_mph")
    dist_mi = trill_metadata.get("distance_miles")
    dur_s = trill_metadata.get("duration_seconds")

    location = f"near {place}, {state}" if place and state else state if state else "the open road"
    if state_usps and state_usps not in location:
        location = f"{location} (USPS {state_usps})".strip()
    scene = (
        f"{protected_name} (verified protected lands)"
        if near_protected and protected_name
        else "public lands"
        if near_protected
        else "backroads"
    )

    prefs = user_prefs or {}
    model = prefs.get("trill_openai_model", "gpt-4o-mini")

    user_prompt = f"""Write title, caption, and hashtags for a driving clip using only these facts (do not invent
scenes or stunts). Sound like a real person posting their footage — no emojis anywhere.

TRILL SCORE: {score}/100 (higher = more thrilling)
SPEED BUCKET: {bucket}
CENSUS NEAREST PLACE (when gazetteer configured on server): {place or "(not resolved)"}
STATE / REGION: {state or "(unknown)"}
LOCATION LINE: {location}
PADUS / PROTECTED LANDS: {scene} (near_protected={near_protected})
ROUTE STATS: peak_speed_mph={max_mph} avg_speed_mph={avg_mph} distance_miles={dist_mi} duration_s={dur_s}
ELEVATION GAIN: {elev_gain}m
CURVATURE: {curv_score}/10 (higher = more twisty/switchbacks)
DYNAMICS: {dyn_score}/10 (higher = more spirited cornering)
MOTION FLAGS: {"Turny roads" if turny else ""} {"Spirited driving" if spirited else ""}

GENERATE:
1. TITLE (max 80 chars)
   - Specific to location/scene/motion flags above; avoid generic AI patterns ("POV:", "Wait for it", "Nobody talks about")
   - No emojis or emoticons; normal capitalization (not Title Case Every Word)

2. CAPTION (max 200 chars)
   - First-person or direct address only if it fits the tone; conversational, not salesy
   - Optional one short question; no FOMO clichés; no emojis

3. HASHTAGS (exactly 15 tags)
   - 3-4 mega viral: #fyp #foryou #viral #trending
   - 4-5 niche community: #roadtrip #driving #explore
   - 3-4 location: #{state} #{place} (if available)
   - 2-3 motion: #curvyroads #spiriteddrive (if applicable)
   - 1-2 protected lands: #publiclands #nationalpark (ONLY if verified: {near_protected})

RETURN ONLY THIS JSON (no markdown, no backticks):
{{
  "title": "your title here",
  "caption": "your caption here",
  "hashtags": ["tag1", "tag2", ...]
}}
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRILL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.85,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        result = json.loads(content)

        hashtags = result.get("hashtags", [])
        hashtags = [h.lower().replace("#", "").replace(" ", "") for h in hashtags]
        hashtags = [h for h in hashtags if h and len(h) <= 30][:15]

        return {
            "title": result.get("title", "")[:100],
            "caption": result.get("caption", "")[:250],
            "hashtags": hashtags,
            "tokens_used": tokens_used,
            "model": model,
            "trill_score": score,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        return {
            "title": trill_metadata.get("title", ""),
            "caption": trill_metadata.get("caption", ""),
            "hashtags": trill_metadata.get("hashtags", []),
            "tokens_used": 0,
            "model": "fallback",
            "error": str(e),
        }
