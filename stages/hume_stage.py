"""
UploadM8 Hume AI Stage
======================
Detect vocal emotion from video audio using Hume AI's Prosody model.
Returns 48 emotional dimensions per speech segment — not just what was
said (Whisper) but HOW it was said (excitement, confidence, anxiety etc.)

Emotion signals drive:
  - Caption voice selection (excitement → hype_street, calm → educational)
  - Thumbnail template mood (high excitement → HEAT/NEON_DROP)
  - Content energy scoring

Free tier: $20 in credits on signup (~300+ minutes of audio).
Then ~$0.064/min.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .context import JobContext
from .errors import SkipStage

logger = logging.getLogger("uploadm8-worker")

HUME_API_KEY   = os.environ.get("HUME_API_KEY", "")
HUME_ENABLED   = os.environ.get("HUME_ENABLED", "true").lower() == "true"
HUME_BASE_URL  = "https://api.hume.ai/v0/batch"
POLL_INTERVAL  = 3.0   # seconds between status checks
MAX_POLLS      = 40    # 40 × 3s = 2 min max wait

# The 48 Hume emotion names (for reference)
HUME_EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness", "Concentration",
    "Confusion", "Contemplation", "Contempt", "Contentment", "Craving",
    "Desire", "Determination", "Disappointment", "Disapproval", "Disgust",
    "Distress", "Doubt", "Ecstasy", "Embarrassment", "Empathic Pain",
    "Enthusiasm", "Entrancement", "Envy", "Excitement", "Fear", "Gratitude",
    "Guilt", "Horror", "Interest", "Joy", "Love", "Nostalgia", "Pain",
    "Pride", "Realization", "Relief", "Romance", "Sadness", "Satisfaction",
    "Shame", "Surprise (positive)", "Surprise (negative)",
]

# Map dominant emotions → caption styles
EMOTION_TO_CAPTION_STYLE = {
    "Excitement":    "hype_street",
    "Enthusiasm":    "hype_street",
    "Amusement":     "humor_casual",
    "Joy":           "humor_casual",
    "Determination": "inspirational_quote",
    "Pride":         "inspirational_quote",
    "Concentration": "educational_informative",
    "Interest":      "educational_informative",
    "Contemplation": "storytelling_narrative",
    "Awe":           "storytelling_narrative",
    "Admiration":    "storytelling_narrative",
    "Anger":         "controversial_bold",
    "Disgust":       "controversial_bold",
}

EMOTION_TO_THUMBNAIL_MOOD = {
    "Excitement":    "bold_dramatic",
    "Enthusiasm":    "bold_dramatic",
    "Amusement":     "bright_energetic",
    "Joy":           "bright_energetic",
    "Determination": "dark_cinematic",
    "Anger":         "bold_dramatic",
    "Concentration": "professional_clean",
    "Interest":      "clean_minimal",
    "Awe":           "dark_cinematic",
}


async def analyze_voice_emotion(audio_path: Path) -> Optional[Dict[str, Any]]:
    """
    Submit audio to Hume AI Batch API and return aggregated emotion scores.
    Returns None if Hume is not configured or analysis fails.
    """
    if not HUME_ENABLED or not HUME_API_KEY:
        return None

    if not audio_path.exists():
        return None

    file_size = audio_path.stat().st_size
    if file_size < 2000:
        logger.info("[hume] Audio too small — skipping emotion analysis")
        return None

    try:
        audio_bytes = audio_path.read_bytes()
        headers = {
            "X-Hume-Api-Key": HUME_API_KEY,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # ── Submit batch job ──────────────────────────────────────────────
            resp = await client.post(
                f"{HUME_BASE_URL}/jobs",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "models": {"prosody": {}},
                    "transcription": {"language": "en"},
                    "notify": False,
                },
            )

            if resp.status_code not in (200, 201):
                logger.warning(f"[hume] Job submit failed: {resp.status_code} {resp.text[:200]}")
                return None

            job_id = resp.json().get("job_id")
            if not job_id:
                logger.warning("[hume] No job_id in response")
                return None

            # ── Upload audio file to the job ──────────────────────────────────
            upload_resp = await client.post(
                f"{HUME_BASE_URL}/jobs/{job_id}/files",
                headers=headers,
                files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
            )

            if upload_resp.status_code not in (200, 201):
                logger.warning(f"[hume] File upload failed: {upload_resp.status_code}")
                return None

            # ── Start the job ─────────────────────────────────────────────────
            start_resp = await client.post(
                f"{HUME_BASE_URL}/jobs/{job_id}/start",
                headers=headers,
            )

            # ── Poll for completion ───────────────────────────────────────────
            for attempt in range(MAX_POLLS):
                await asyncio.sleep(POLL_INTERVAL)

                status_resp = await client.get(
                    f"{HUME_BASE_URL}/jobs/{job_id}",
                    headers=headers,
                )

                if status_resp.status_code != 200:
                    continue

                state = status_resp.json().get("state", {})
                status = state.get("status", "")

                if status == "COMPLETED":
                    break
                elif status in ("FAILED", "CANCELLED"):
                    logger.warning(f"[hume] Job {status}: {state}")
                    return None

            else:
                logger.warning("[hume] Job timed out after polling")
                return None

            # ── Fetch predictions ─────────────────────────────────────────────
            pred_resp = await client.get(
                f"{HUME_BASE_URL}/jobs/{job_id}/predictions",
                headers=headers,
            )

            if pred_resp.status_code != 200:
                logger.warning(f"[hume] Predictions fetch failed: {pred_resp.status_code}")
                return None

            return _parse_hume_predictions(pred_resp.json())

    except Exception as e:
        logger.warning(f"[hume] Analysis error: {e}")
        return None


def _parse_hume_predictions(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Parse Hume batch predictions response.
    Aggregates emotion scores across all speech segments.
    Returns structured emotion context dict.
    """
    try:
        all_segments: List[Dict] = []
        emotion_totals: Dict[str, float] = {}
        emotion_counts: Dict[str, int] = {}

        # Navigate: predictions list → results → prosody → grouped_predictions → predictions
        for file_pred in raw:
            results = file_pred.get("results", {}).get("predictions", [])
            for result in results:
                prosody = result.get("models", {}).get("prosody", {})
                for group in prosody.get("grouped_predictions", []):
                    for seg in group.get("predictions", []):
                        emotions = seg.get("emotions", [])
                        for emo in emotions:
                            name  = emo.get("name", "")
                            score = float(emo.get("score", 0))
                            emotion_totals[name] = emotion_totals.get(name, 0) + score
                            emotion_counts[name] = emotion_counts.get(name, 0) + 1

                        all_segments.append({
                            "start": seg.get("time", {}).get("begin", 0),
                            "end":   seg.get("time", {}).get("end", 0),
                            "text":  seg.get("text", ""),
                        })

        if not emotion_totals:
            return None

        # Average across segments
        avg_scores = {
            name: emotion_totals[name] / emotion_counts[name]
            for name in emotion_totals
        }

        # Sort by score
        sorted_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = [{"name": n, "score": round(s, 4)} for n, s in sorted_emotions[:5]]
        dominant = sorted_emotions[0][0] if sorted_emotions else "Interest"

        # Map to caption style + thumbnail mood
        caption_style    = EMOTION_TO_CAPTION_STYLE.get(dominant, "hype_street")
        thumbnail_mood   = EMOTION_TO_THUMBNAIL_MOOD.get(dominant, "bold_dramatic")
        dominant_score   = avg_scores.get(dominant, 0)

        # Emotional intensity (0–1 scale)
        high_energy_emotions = ["Excitement", "Enthusiasm", "Joy", "Anger", "Ecstasy", "Amusement"]
        intensity = "high" if dominant in high_energy_emotions and dominant_score > 0.3 else "moderate"

        logger.info(
            f"[hume] Dominant emotion: {dominant} ({dominant_score:.3f}) "
            f"→ style={caption_style} mood={thumbnail_mood}"
        )

        return {
            "dominant_emotion":   dominant,
            "dominant_score":     round(dominant_score, 4),
            "top_emotions":       top_5,
            "all_scores":         {n: round(s, 4) for n, s in sorted_emotions[:15]},
            "caption_style_hint": caption_style,
            "thumbnail_mood_hint": thumbnail_mood,
            "emotional_intensity": intensity,
            "segment_count":      len(all_segments),
        }

    except Exception as e:
        logger.warning(f"[hume] Parse error: {e}")
        return None


def merge_hume_into_audio_context(
    audio_ctx: Dict[str, Any],
    hume_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge Hume emotion data into existing audio_context dict.
    Hume overrides GPT-derived caption_style and thumbnail_mood
    when it has high confidence (dominant_score > 0.25).
    """
    if not hume_result:
        return audio_ctx

    audio_ctx["hume_emotions"] = hume_result

    dominant_score = hume_result.get("dominant_score", 0)

    # Override GPT classification with Hume data when confidence is high
    if dominant_score > 0.25:
        audio_ctx["caption_style"]   = hume_result["caption_style_hint"]
        audio_ctx["thumbnail_mood"]  = hume_result["thumbnail_mood_hint"]
        audio_ctx["emotional_tone"]  = _map_dominant_to_tone(hume_result["dominant_emotion"])
        logger.info(
            f"[hume] Overriding caption_style={audio_ctx['caption_style']} "
            f"thumbnail_mood={audio_ctx['thumbnail_mood']}"
        )

    return audio_ctx


def _map_dominant_to_tone(dominant: str) -> str:
    """Map Hume dominant emotion to UploadM8 emotional_tone enum."""
    mapping = {
        "Excitement":    "hype_energetic",
        "Enthusiasm":    "hype_energetic",
        "Joy":           "funny_playful",
        "Amusement":     "funny_playful",
        "Determination": "inspirational_motivational",
        "Pride":         "inspirational_motivational",
        "Concentration": "educational_informative",
        "Interest":      "educational_informative",
        "Contemplation": "calm_relaxed",
        "Calmness":      "calm_relaxed",
        "Awe":           "dramatic_intense",
        "Horror":        "dramatic_intense",
        "Anger":         "controversial_bold",
        "Disgust":       "controversial_bold",
        "Sadness":       "romantic_emotional",
        "Love":          "romantic_emotional",
        "Nostalgia":     "romantic_emotional",
    }
    return mapping.get(dominant, "hype_energetic")
