"""
UploadM8 Audio Stage — FULL STACK (Stage 5.5)
==============================================
Multi-modal audio intelligence pipeline. Runs before thumbnail + caption.

Stack:
  1. FFmpeg       — extract mono MP3 audio clip
  2. Whisper      — speech-to-text transcription (OpenAI API)
  3. ACRCloud     — music/song recognition + copyright detection
  4. YAMNet       — 521 AudioSet sound event classification (free, local)
  5. Hume AI      — 48-dimension voice emotion detection
  6. GPT-4o-mini  — content classification (18 categories + thumbnail/caption directives)

All signals merged into ctx.audio_context dict consumed downstream.
Also populates ctx.ai_transcript for caption_stage and ctx.audio_path for cleanup.

Pipeline position:
    transcode_stage  →  [audio_context_stage]  →  thumbnail_stage  →  caption_stage

Graceful failure guarantee:
  Every sub-stage is non-fatal — pipeline continues on any individual failure.
  On total failure, ctx.audio_context = _empty_context(), ctx.ai_transcript = None.

Environment variables:
  OPENAI_API_KEY                required for Whisper
  AUDIO_STAGE_ENABLED           default: true
  YAMNET_ENABLED                default: true
  HUME_ENABLED                  default: true
  MAX_AUDIO_DURATION_SECONDS    default: 120 (clip length for extraction)
  FFMPEG_PATH                   default: ffmpeg
  ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, ACRCLOUD_ACCESS_SECRET  (optional)
  WHISPER_MODEL                 default: whisper-1
  AUDIO_TRANSCRIPT_MAX_CHARS    default: 3000
  AUDIO_MIN_DURATION_SECS       default: 3.0
  AUDIO_MAX_DURATION_SECS       default: 600.0
  FFMPEG_AUDIO_TIMEOUT          default: 120

User opt-out:
  ctx.user_settings["use_audio_context"] = False  →  SkipStage (entire pipeline)
  ctx.user_settings["audio_transcription"] / audioTranscription = False  →  Skip Whisper only;
    YAMNet, ACRCloud (if configured), Hume, GPT classification still run.

Exports: run_audio_context_stage(ctx)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .errors import AudioError, SkipStage
from .context import JobContext
from .outbound_rl import outbound_slot

logger = logging.getLogger("uploadm8-worker.audio")

# ── Environment-tunable constants ──────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "whisper-1")
TRANSCRIPT_MAX_CHARS: int = int(os.environ.get("AUDIO_TRANSCRIPT_MAX_CHARS", "3000"))
AUDIO_MIN_DURATION: float = float(os.environ.get("AUDIO_MIN_DURATION_SECS", "3.0"))
AUDIO_MAX_DURATION: float = float(os.environ.get("AUDIO_MAX_DURATION_SECS", "600.0"))
FFMPEG_TIMEOUT: int = int(os.environ.get("FFMPEG_AUDIO_TIMEOUT", "120"))

# Full-stack audio pipeline
FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH", "ffmpeg")
# ACRCloud: accept docs-style ACR_* aliases alongside ACRCLOUD_*
ACRCLOUD_HOST: str = os.environ.get("ACRCLOUD_HOST") or os.environ.get("ACR_HOST", "")
ACRCLOUD_ACCESS_KEY: str = os.environ.get("ACRCLOUD_ACCESS_KEY") or os.environ.get("ACR_ACCESS_KEY", "")
ACRCLOUD_ACCESS_SECRET: str = os.environ.get("ACRCLOUD_ACCESS_SECRET") or os.environ.get("ACR_ACCESS_SECRET", "")
AUDIO_STAGE_ENABLED: bool = os.environ.get("AUDIO_STAGE_ENABLED", "true").lower() == "true"
YAMNET_ENABLED: bool = os.environ.get("YAMNET_ENABLED", "true").lower() == "true"
HUME_ENABLED_FLAG: bool = os.environ.get("HUME_ENABLED", "true").lower() == "true"
MAX_AUDIO_SECONDS: int = int(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "120"))

CATEGORIES = ["automotive", "sports_extreme", "gaming", "music_performance", "food_cooking", "travel_vlog", "fitness_workout", "comedy_entertainment", "educational", "lifestyle_fashion", "pets_animals", "nature_outdoors", "business_finance", "technology", "art_creative", "family_kids", "news_commentary", "other"]
EMOTIONAL_TONES = ["hype_energetic", "calm_relaxed", "funny_playful", "educational_informative", "dramatic_intense", "inspirational_motivational", "romantic_emotional", "controversial_bold"]
THUMBNAIL_MOODS = ["bold_dramatic", "clean_minimal", "neon_vibrant", "dark_cinematic", "bright_energetic", "professional_clean"]
CAPTION_STYLES = ["hype_street", "educational_informative", "storytelling_narrative", "question_hook", "challenge_call_to_action", "humor_casual", "inspirational_quote"]

# ── Whisper API endpoint ───────────────────────────────────────────────────
WHISPER_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_info_get(ctx: JobContext, field: str, default: Any = None) -> Any:
    """
    Read a field from ctx.video_info regardless of whether it is stored as a
    dict (the declared type) or as a VideoInfo dataclass (what transcode_stage
    actually produces in some code paths).
    """
    vi = ctx.video_info
    if vi is None:
        return default
    if isinstance(vi, dict):
        return vi.get(field, default)
    # VideoInfo dataclass / namedtuple / any other object
    return getattr(vi, field, default)


def _get_video_source(ctx: JobContext) -> Optional[Path]:
    """
    Return the best available video path for audio extraction.
    Prefer the transcoded output; fall back to the local original.
    """
    # Prefer the first available platform video (all share the same audio track)
    if ctx.platform_videos:
        for p in ctx.platform_videos.values():
            if p and Path(p).exists():
                return Path(p)

    if ctx.processed_video_path and Path(ctx.processed_video_path).exists():
        return Path(ctx.processed_video_path)

    if ctx.local_video_path and Path(ctx.local_video_path).exists():
        return Path(ctx.local_video_path)

    return None


# ---------------------------------------------------------------------------
# FFmpeg audio extraction (MP3 for full stack — Whisper/ACRCloud/Hume accept it)
# ---------------------------------------------------------------------------

async def _noop() -> None:
    return None


async def _extract_audio_clip(video_path: Path, temp_dir: Path) -> Optional[Path]:
    """Extract mono MP3 clip (max MAX_AUDIO_SECONDS) for full-stack analysis."""
    output_path = temp_dir / "audio_analysis.mp3"
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-ab", "64k",
        "-ar", "16000", "-ac", "1", "-t", str(MAX_AUDIO_SECONDS),
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode == 0 and output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            if size_kb < 1:
                return None
            logger.info(f"[audio] Audio extracted: {size_kb:.1f} KB")
            return output_path
        logger.warning(f"[audio] FFmpeg extract failed: {stderr.decode()[:200]}")
        return None
    except Exception as e:
        logger.warning(f"[audio] Extraction error: {e}")
        return None


async def _extract_audio_wav(video_path: Path, output_path: Path) -> bool:
    """
    Extract mono 16kHz WAV from video using FFmpeg.

    Args:
        video_path:  source video file
        output_path: destination WAV path (inside ctx.temp_dir)

    Returns:
        True on success, False on any failure.
    """
    cmd = [
        "ffmpeg",
        "-y",                          # overwrite if exists
        "-i", str(video_path),
        "-vn",                         # no video
        "-acodec", "pcm_s16le",        # 16-bit linear PCM
        "-ar", "16000",                # 16 kHz — Whisper-optimal
        "-ac", "1",                    # mono
        str(output_path),
    ]

    logger.debug(f"FFmpeg audio extract: {' '.join(cmd[:6])} ...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=FFMPEG_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning(
                f"FFmpeg audio extraction timed out after {FFMPEG_TIMEOUT}s "
                f"for {video_path.name}"
            )
            return False

        if proc.returncode != 0:
            snippet = stderr.decode("utf-8", errors="replace")[-300:]
            logger.warning(
                f"FFmpeg audio extraction failed (rc={proc.returncode}): {snippet}"
            )
            return False

        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.warning(
                f"FFmpeg produced an empty/missing WAV for {video_path.name}"
            )
            return False

        size_kb = output_path.stat().st_size / 1024
        logger.debug(f"WAV extracted: {output_path.name} ({size_kb:.1f} KB)")
        return True

    except FileNotFoundError:
        # ffmpeg not on PATH
        logger.warning("FFmpeg not found on PATH — audio context stage cannot run")
        return False
    except Exception as e:
        logger.warning(f"FFmpeg audio extraction exception: {e}")
        return False


# ---------------------------------------------------------------------------
# YAMNet audio classification (super charge, no account)
# ---------------------------------------------------------------------------

def _classify_audio_yamnet(wav_path: Path) -> Optional[list]:
    """
    Classify audio using YAMNet via TensorFlow Hub. Local, no API key.
    First run downloads ~20MB model. Returns top-5 class names or None.
    """
    try:
        import csv
        import tensorflow as tf
        import tensorflow_hub as hub
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    try:
        waveform, sample_rate = sf.read(wav_path)
        if sample_rate != 16000:
            import resampy
            waveform = resampy.resample(waveform, sample_rate, 16000)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32) / 32768.0
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        scores, _, _ = model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top5_idx = np.argsort(mean_scores)[-5:][::-1]
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        class_names = []
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row["display_name"])
        return [class_names[i] for i in top5_idx if i < len(class_names)]
    except Exception as e:
        logger.debug(f"YAMNet audio classification failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

async def _call_whisper(wav_path: Path) -> Optional[str]:
    """
    Send a WAV file to OpenAI Whisper and return the transcript text.

    Uses httpx (same pattern as caption_stage) to avoid SDK dependency.

    Returns:
        Transcript string (may be empty) on success, None on any failure.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — cannot call Whisper")
        return None

    try:
        wav_bytes = wav_path.read_bytes()
    except Exception as e:
        logger.warning(f"Could not read WAV file: {e}")
        return None

    try:
        async with outbound_slot("openai"):
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    WHISPER_ENDPOINT,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files={"file": (wav_path.name, wav_bytes, "audio/wav")},
                    data={"model": WHISPER_MODEL},
                )

            if resp.status_code == 429:
                logger.warning("Whisper API rate limited (429) — skipping transcript")
                return None

            if resp.status_code == 400:
                # Usually empty audio / unsupported format
                body = resp.text[:200]
                logger.warning(f"Whisper API bad request (400): {body}")
                return None

            if resp.status_code != 200:
                body = resp.text[:200]
                logger.warning(f"Whisper API error {resp.status_code}: {body}")
                return None

            data = resp.json()
            text = data.get("text", "").strip()
            return text

    except httpx.TimeoutException:
        logger.warning("Whisper API request timed out")
        return None
    except Exception as e:
        logger.warning(f"Whisper API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Full-stack: Whisper verbose_json (for audio_context)
# ---------------------------------------------------------------------------

async def _transcribe_audio(audio_path: Path) -> Optional[Dict[str, Any]]:
    """Whisper with verbose_json — returns text, language, duration, segments."""
    try:
        if audio_path.stat().st_size < 2_000 or audio_path.stat().st_size > 24 * 1024 * 1024:
            return None
        audio_bytes = audio_path.read_bytes()
        mime = "audio/mpeg" if str(audio_path).lower().endswith(".mp3") else "audio/wav"
        async with outbound_slot("openai"):
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    WHISPER_ENDPOINT,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files={"file": ("audio.mp3", audio_bytes, mime)},
                    data={"model": WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0"},
                )
        if response.status_code == 200:
            data = response.json()
            text = (data.get("text") or "").strip()
            logger.info(f"[audio] Transcription: {len(text)} chars, lang={data.get('language','?')}")
            return {
                "text": text,
                "language": data.get("language", "en"),
                "duration": data.get("duration", 0),
                "segments": data.get("segments", [])[:15],
            }
        return None
    except Exception as e:
        logger.warning(f"[audio] Whisper error: {e}")
        return None


# ---------------------------------------------------------------------------
# ACRCloud music recognition
# ---------------------------------------------------------------------------

async def _recognize_music(audio_path: Path) -> Optional[Dict[str, Any]]:
    """ACRCloud identify — returns detected track + copyright_warning."""
    if not (ACRCLOUD_HOST and ACRCLOUD_ACCESS_KEY and ACRCLOUD_ACCESS_SECRET):
        return None
    try:
        timestamp = str(int(time.time()))
        string_to_sign = "\n".join(["POST", "/v1/identify", ACRCLOUD_ACCESS_KEY, "audio", "1", timestamp])
        signature = base64.b64encode(
            hmac.new(ACRCLOUD_ACCESS_SECRET.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        ).decode("utf-8")
        sample = audio_path.read_bytes()[:512 * 1024]
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://{ACRCLOUD_HOST}/v1/identify",
                data={
                    "access_key": ACRCLOUD_ACCESS_KEY,
                    "timestamp": timestamp,
                    "signature": signature,
                    "data_type": "audio",
                    "signature_version": "1",
                    "sample_bytes": str(len(sample)),
                },
                files={"sample": ("audio.mp3", sample, "audio/mpeg")},
            )
        if response.status_code != 200:
            return {"detected": False, "copyright_warning": False}
        data = response.json()
        if data.get("status", {}).get("code") == 0:
            music_list = data.get("metadata", {}).get("music", [])
            if music_list:
                track = music_list[0]
                artist = track["artists"][0]["name"] if track.get("artists") else ""
                genre = track["genres"][0]["name"] if track.get("genres") else ""
                logger.info(f"[audio] Music detected: '{track.get('title')}' by '{artist}'")
                return {
                    "detected": True,
                    "title": track.get("title", ""),
                    "artist": artist,
                    "album": track.get("album", {}).get("name", ""),
                    "genre": genre,
                    "release_date": track.get("release_date", ""),
                    "copyright_warning": True,
                }
        return {"detected": False, "copyright_warning": False}
    except Exception as e:
        logger.warning(f"[audio] ACRCloud error: {e}")
        return None


# ---------------------------------------------------------------------------
# GPT content classification
# ---------------------------------------------------------------------------

async def _analyze_audio_context(
    transcript: Optional[Dict[str, Any]],
    music_info: Optional[Dict[str, Any]],
    filename: str,
    platforms: List[str],
    yamnet_context: Dict[str, Any],
) -> Dict[str, Any]:
    """GPT-4o-mini classification — category, emotion, mood, caption_style, etc."""
    transcript_text = (transcript or {}).get("text", "")
    language = (transcript or {}).get("language", "en")
    music_detected = (music_info or {}).get("detected", False)
    music_genre = (music_info or {}).get("genre", "")
    music_title = (music_info or {}).get("title", "")
    music_artist = (music_info or {}).get("artist", "")
    copyright_flag = (music_info or {}).get("copyright_warning", False)
    sound_profile = yamnet_context.get("sound_profile", "")
    yamnet_top = yamnet_context.get("top_sound_class", "")

    analysis: Dict[str, Any] = {}

    if OPENAI_API_KEY:
        analysis_input = {
            "filename": filename,
            "transcript_excerpt": transcript_text[:600],
            "has_speech": bool(transcript_text),
            "music_detected": music_detected,
            "music_title": music_title,
            "music_artist": music_artist,
            "music_genre": music_genre,
            "yamnet_sound_profile": sound_profile,
            "yamnet_top_class": yamnet_top,
            "platforms": platforms,
        }
        prompt = f"""You are a viral content strategist. Analyse the following video metadata and audio signals.

CRITICAL: The transcript may be SONG LYRICS from a third-party track (especially when music_detected is true
or a known title/artist is present). In that case the creator is usually filming a SCENE (drive, workout, etc.),
not claiming they wrote the song. Set transcript_role accordingly — do not treat lyrics as the creator's own words.

Valid transcript_role values:
- third_party_lyrics — transcript is mostly recognizable song lyrics / performed vocal track
- creator_speech — creator talking to camera or narrating (original speech)
- mixed_speech_and_music — both speech and prominent music
- ambient_or_unclear — no clear lyrics, ambient, or unclear

Input: {json.dumps(analysis_input, indent=2)}
Valid categories: {", ".join(CATEGORIES)}
Valid emotional_tone: {", ".join(EMOTIONAL_TONES)}
Valid thumbnail_mood: {", ".join(THUMBNAIL_MOODS)}
Valid caption_style: {", ".join(CAPTION_STYLES)}
Return ONLY valid JSON, no markdown:
{{"category":"<cat>","subcategory":"<niche>","transcript_role":"<transcript_role>","fusion_narrative":"<one sentence describing what is actually going on>","emotional_tone":"<tone>","content_signals":["sig1"],"caption_style":"<style>","target_audience":"<audience>","thumbnail_mood":"<mood>","copyright_risk":false,"suggested_keywords":["kw1"],"thumbnail_headline":"VIRAL HEADLINE","thumbnail_subtext":"optional subtext"}}"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 600,
                        "temperature": 0.4,
                        "response_format": {"type": "json_object"},
                    },
                )
            if response.status_code == 200:
                analysis = json.loads(response.json()["choices"][0]["message"]["content"])
                logger.info(f"[audio] GPT: {analysis.get('category')}/{analysis.get('subcategory')}")
        except Exception as e:
            logger.warning(f"[audio] GPT error: {e}")

    tr_role = (analysis.get("transcript_role") or "").strip()
    if not tr_role and transcript_text and music_detected and (music_title or music_artist):
        tr_role = "third_party_lyrics"
    if not tr_role and transcript_text and music_detected and copyright_flag:
        tr_role = "third_party_lyrics"

    return {
        "transcript": transcript_text,
        "language": language,
        "transcript_segments": (transcript or {}).get("segments", []),
        "music_detected": music_detected,
        "music_title": music_title,
        "music_artist": music_artist,
        "music_genre": music_genre,
        "copyright_risk": copyright_flag or bool(analysis.get("copyright_risk", False)),
        "category": analysis.get("category", "other"),
        "subcategory": analysis.get("subcategory", ""),
        "transcript_role": tr_role,
        "fusion_narrative": (analysis.get("fusion_narrative") or "").strip(),
        "emotional_tone": analysis.get("emotional_tone", "hype_energetic"),
        "content_signals": analysis.get("content_signals", []),
        "caption_style": analysis.get("caption_style", "hype_street"),
        "target_audience": analysis.get("target_audience", ""),
        "thumbnail_mood": analysis.get("thumbnail_mood", "bold_dramatic"),
        "suggested_keywords": analysis.get("suggested_keywords", []),
        "thumbnail_headline": analysis.get("thumbnail_headline", ""),
        "thumbnail_subtext": analysis.get("thumbnail_subtext", ""),
    }


def _empty_context() -> Dict[str, Any]:
    return {
        "transcript": "",
        "language": "en",
        "transcript_segments": [],
        "music_detected": False,
        "music_title": "",
        "music_artist": "",
        "music_genre": "",
        "copyright_risk": False,
        "category": "other",
        "subcategory": "",
        "transcript_role": "",
        "fusion_narrative": "",
        "emotional_tone": "hype_energetic",
        "content_signals": [],
        "caption_style": "hype_street",
        "target_audience": "",
        "thumbnail_mood": "bold_dramatic",
        "suggested_keywords": [],
        "thumbnail_headline": "",
        "thumbnail_subtext": "",
        "yamnet_events": [],
        "sound_profile": "unknown",
    }


def _yamnet_to_context(labels: Optional[List[str]]) -> Dict[str, Any]:
    """Build yamnet_context from inline _classify_audio_yamnet result."""
    if not labels:
        return {"sound_profile": "unknown", "top_sound_class": "", "yamnet_events": []}
    return {
        "sound_profile": ", ".join(labels[:5]) if labels else "unknown",
        "top_sound_class": labels[0] if labels else "unknown",
        "yamnet_events": labels or [],
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

async def run_audio_context_stage(ctx: JobContext) -> JobContext:
    """
    Stage 5.5 — Full-stack audio intelligence pipeline.

    Populates:
        ctx.audio_context  (dict) — category, emotion, mood, transcript, etc.
        ctx.ai_transcript  (str | None) — capped transcript for caption_stage
        ctx.audio_path     (Path | None) — temp audio path, cleaned with temp_dir

    Raises:
        SkipStage   — clean skip (disabled, no API key, no audio, etc.)
    """
    ctx.mark_stage("audio")

    if not AUDIO_STAGE_ENABLED:
        raise SkipStage("Audio stage disabled via env")

    if not OPENAI_API_KEY:
        raise SkipStage("OPENAI_API_KEY not configured — audio context disabled")

    use_audio = (ctx.user_settings or {}).get("use_audio_context", True)
    if use_audio is None:
        use_audio = (ctx.user_settings or {}).get("useAudioContext", True)
    if not use_audio:
        raise SkipStage("Audio context disabled by user preference")

    us = ctx.user_settings or {}
    use_whisper = us.get("audio_transcription")
    if use_whisper is None:
        use_whisper = us.get("audioTranscription", True)
    use_whisper = bool(use_whisper)

    audio_codec = _video_info_get(ctx, "audio_codec")
    if not audio_codec:
        raise SkipStage("Video has no audio stream")

    duration = _video_info_get(ctx, "duration", 0.0)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    if duration < AUDIO_MIN_DURATION:
        raise SkipStage(
            f"Clip too short for transcription ({duration:.1f}s < {AUDIO_MIN_DURATION}s minimum)"
        )

    if duration > AUDIO_MAX_DURATION:
        raise SkipStage(
            f"Clip too long for transcription ({duration:.1f}s > {AUDIO_MAX_DURATION}s cap)"
        )

    video_path = _get_video_source(ctx)
    if not video_path or not video_path.exists():
        raise SkipStage("No video file available for audio extraction")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory — cannot write audio file")

    temp_dir = Path(ctx.temp_dir)

    logger.info(
        f"Audio context: {ctx.filename or ctx.upload_id} | "
        f"duration={duration:.1f}s | codec={audio_codec}"
    )

    try:
        audio_path = await _extract_audio_clip(video_path, temp_dir)
        if not audio_path or not audio_path.exists():
            ctx.audio_context = _empty_context()
            return ctx

        ctx.audio_path = audio_path

        async def _whisper_or_skip() -> Optional[Dict[str, Any]]:
            if not use_whisper:
                logger.info("[audio] Whisper skipped — audio transcription disabled in user preferences")
                return None
            return await _transcribe_audio(audio_path)

        # Run Whisper + ACRCloud in parallel (Whisper optional per user pref)
        transcript_task = asyncio.create_task(_whisper_or_skip())
        acrcloud_task = asyncio.create_task(
            _recognize_music(audio_path)
            if (ACRCLOUD_HOST and ACRCLOUD_ACCESS_KEY and ACRCLOUD_ACCESS_SECRET)
            else _noop()
        )

        transcript_result, music_result = await asyncio.gather(
            transcript_task, acrcloud_task, return_exceptions=True
        )
        if isinstance(transcript_result, Exception):
            logger.warning(f"[audio] Transcription error: {transcript_result}")
            transcript_result = None
        if isinstance(music_result, Exception):
            logger.warning(f"[audio] ACRCloud error: {music_result}")
            music_result = None

        # YAMNet sound classification
        yamnet_context: Dict[str, Any] = {}
        if YAMNET_ENABLED:
            try:
                from .yamnet_stage import run_yamnet_classification, yamnet_to_context
                yamnet_events = await run_yamnet_classification(audio_path)
                yamnet_context = yamnet_to_context(yamnet_events)
            except ImportError:
                # Fallback: inline YAMNet when yamnet_stage not present
                loop = asyncio.get_event_loop()
                labels = await loop.run_in_executor(None, _classify_audio_yamnet, audio_path)
                yamnet_context = _yamnet_to_context(labels)
            except Exception as e:
                logger.warning(f"[audio] YAMNet error (non-fatal): {e}")

        # GPT content classification
        audio_context = await _analyze_audio_context(
            transcript=transcript_result,
            music_info=music_result,
            filename=ctx.filename,
            platforms=ctx.platforms or [],
            yamnet_context=yamnet_context,
        )
        audio_context.update(yamnet_context)

        # Hume AI voice emotion
        if HUME_ENABLED_FLAG:
            try:
                from .hume_stage import analyze_voice_emotion, merge_hume_into_audio_context
                hume_result = await analyze_voice_emotion(audio_path)
                audio_context = merge_hume_into_audio_context(audio_context, hume_result)
            except Exception as e:
                logger.warning(f"[audio] Hume error (non-fatal): {e}")

        ctx.audio_context = audio_context

        # Populate ai_transcript for caption_stage
        transcript_text = (audio_context.get("transcript") or "").strip()
        if transcript_text:
            if len(transcript_text) > TRANSCRIPT_MAX_CHARS:
                transcript_text = transcript_text[:TRANSCRIPT_MAX_CHARS]
            ctx.ai_transcript = transcript_text

        logger.info(
            f"[audio]  category={audio_context.get('category')} "
            f"emotion={audio_context.get('emotional_tone')} "
            f"mood={audio_context.get('thumbnail_mood')} "
            f"yamnet={audio_context.get('top_sound_class', 'N/A')} "
            f"hume={audio_context.get('hume_emotions', {}).get('dominant_emotion', 'N/A')}"
        )
        return ctx

    except SkipStage:
        raise
    except Exception as e:
        logger.warning(f"[audio] Non-fatal stage error: {e}")
        ctx.audio_context = _empty_context()
        return ctx
