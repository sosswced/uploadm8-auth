"""
UploadM8 Audio Context Stage
==============================
Extract audio from video and transcribe using OpenAI Whisper.
Adds ctx.ai_transcript for the thumbnail and caption stages to use as
grounding evidence.

Flow:
  1. Check entitlements + user setting (use_audio_context)
  2. Skip if no audio stream in video_info
  3. Extract 16 kHz mono WAV using FFmpeg (-vn)
  4. Call Whisper API (multipart POST to /v1/audio/transcriptions)
  5. Store transcript in ctx.ai_transcript (capped to TRANSCRIPT_MAX_CHARS)
  6. Temp WAV is cleaned up automatically with ctx.temp_dir

Placement: After transcode_stage, BEFORE thumbnail_stage AND caption_stage.
           Running before thumbnail means the transcript is available as
           additional context for any future AI-assisted thumbnail selection
           or frame-picking logic.

Graceful: Any failure at any step → log warning + return ctx unchanged.
          Pipeline always continues; captions just won't have audio context.

Cost reference (whisper-1 at $0.006/min):
  30s clip  → ~$0.003
  60s clip  → ~$0.006
  90s clip  → ~$0.009
  10min cap → ~$0.060 max

Exports: run_audio_context_stage(ctx)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from .errors import SkipStage
from .context import JobContext
from .ai_service_costs import billing_env_from_os, user_pref_ai_service_enabled

logger = logging.getLogger("uploadm8-worker.audio")

# ---------------------------------------------------------------------------
# Configuration (all override-able via environment variables)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")

# Transcript is injected into the caption prompt — cap it to control token spend
TRANSCRIPT_MAX_CHARS = int(os.environ.get("AUDIO_TRANSCRIPT_MAX_CHARS", "3000"))

# Duration guards — skip clips that are too short to be useful or too long to be cheap
AUDIO_MIN_DURATION_SECS = float(os.environ.get("AUDIO_MIN_DURATION_SECS", "3.0"))
AUDIO_MAX_DURATION_SECS = float(os.environ.get("AUDIO_MAX_DURATION_SECS", "600.0"))  # 10 min

# FFmpeg extraction timeout (seconds)
FFMPEG_AUDIO_TIMEOUT = int(os.environ.get("FFMPEG_AUDIO_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# FFmpeg audio extraction
# ---------------------------------------------------------------------------

async def _extract_audio_wav(video_path: Path, out_path: Path) -> bool:
    """
    Extract audio track from video as 16 kHz mono PCM WAV.

    Whisper works best with 16 kHz mono input — this avoids any
    server-side resampling and keeps the WAV size minimal.

    Returns True on success, False on any failure.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                  # strip video stream completely
        "-acodec", "pcm_s16le", # raw PCM, 16-bit little-endian
        "-ar", "16000",         # 16 kHz — Whisper native
        "-ac", "1",             # mono (halves file size vs stereo)
        str(out_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=FFMPEG_AUDIO_TIMEOUT
        )

        if proc.returncode != 0:
            err_tail = stderr_bytes.decode(errors="replace")[-600:]
            logger.warning(f"FFmpeg audio extraction failed (rc={proc.returncode}): {err_tail}")
            return False

        if not out_path.exists() or out_path.stat().st_size < 100:
            logger.warning("FFmpeg produced empty or missing WAV — no audio to transcribe")
            return False

        size_kb = out_path.stat().st_size / 1024
        logger.info(f"Audio extracted: {out_path.name} ({size_kb:.1f} KB)")
        return True

    except asyncio.TimeoutError:
        logger.warning(f"FFmpeg audio extraction timed out after {FFMPEG_AUDIO_TIMEOUT}s")
        try:
            proc.kill()
        except Exception:
            pass
        return False

    except Exception as exc:
        logger.warning(f"FFmpeg audio extraction unexpected error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

async def _gpt_audio_summary_from_transcript(transcript: str) -> Optional[str]:
    """Small GPT-4o-mini summary when aiServiceAudioSummary is on."""
    if not transcript.strip() or not OPENAI_API_KEY:
        return None
    model = os.environ.get("OPENAI_AUDIO_SUMMARY_MODEL", "gpt-4o-mini")
    prompt = (
        "In 2–4 short phrases, summarize themes, pacing, and vibe of this spoken content "
        "for social video metadata. No bullet labels, no JSON.\n\n"
        + transcript[:2500]
    )
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.35,
                },
            )
            if r.status_code != 200:
                logger.warning("GPT audio summary HTTP %s", r.status_code)
                return None
            data = r.json()
            txt = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            return txt.strip() or None
    except Exception as e:
        logger.warning("GPT audio summary failed: %s", e)
        return None


async def _transcribe_wav(wav_path: Path) -> Optional[str]:
    """
    Send a WAV file to OpenAI Whisper and return the transcript text.

    Uses httpx (already a project dependency via caption_stage) with a
    multipart/form-data upload — same pattern as caption_stage.

    Returns transcript string or None on any failure.
    """
    if not OPENAI_API_KEY:
        logger.debug("OPENAI_API_KEY not set — cannot call Whisper")
        return None

    try:
        import httpx

        file_size_kb = wav_path.stat().st_size / 1024
        logger.info(
            f"Sending {file_size_kb:.1f} KB WAV to Whisper API (model={WHISPER_MODEL})"
        )

        async with httpx.AsyncClient(timeout=120) as client:
            with open(wav_path, "rb") as audio_file:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    data={"model": WHISPER_MODEL},
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                )

        if response.status_code == 429:
            logger.warning("Whisper API rate limited — skipping audio context for this job")
            return None

        if response.status_code != 200:
            logger.warning(
                f"Whisper API returned {response.status_code}: {response.text[:400]}"
            )
            return None

        payload = response.json()
        transcript = payload.get("text", "").strip()

        if not transcript:
            logger.info("Whisper returned an empty transcript")
            return None

        logger.info(
            f"Whisper transcript received ({len(transcript)} chars): "
            f"{transcript[:120]}{'...' if len(transcript) > 120 else ''}"
        )
        return transcript

    except Exception as exc:
        logger.warning(f"Whisper transcription failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

async def run_audio_context_stage(ctx: JobContext) -> JobContext:
    """
    Audio analysis stack: Whisper (optional), GPT audio summary (optional),
    plus reserved hooks for YAMNet / ACR when those engines are wired.

    Respects per-service upload preferences (aiService* + audio_transcription).
    """
    ctx.mark_stage("audio")

    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI not enabled for this tier — skipping audio context")

    us = ctx.user_settings or {}
    use_audio = bool(us.get("use_audio_context", us.get("useAudioContext", True)))
    if not use_audio:
        raise SkipStage("Audio context disabled by user setting (use_audio_context=false)")

    env_audio = billing_env_from_os()
    transcribe_pref = bool(us.get("audio_transcription", us.get("audioTranscription", True)))
    want_whisper = transcribe_pref and user_pref_ai_service_enabled(us, "audio_whisper", True)
    want_yamnet = env_audio.get("YAMNET_ENABLED", True) and user_pref_ai_service_enabled(
        us, "audio_yamnet", True
    )
    want_acr = env_audio.get("ACRCLOUD_CONFIGURED", False) and user_pref_ai_service_enabled(
        us, "audio_acr", True
    )
    want_gpt_summary = user_pref_ai_service_enabled(us, "audio_gpt_classify", True)

    if not any((want_whisper, want_yamnet, want_acr, want_gpt_summary)):
        raise SkipStage("All audio analysis services disabled in upload preferences")

    audio_codec = (ctx.video_info or {}).get("audio_codec")
    if not audio_codec:
        raise SkipStage("Video has no audio stream — skipping audio context")

    duration = float((ctx.video_info or {}).get("duration", 0.0))
    if duration < AUDIO_MIN_DURATION_SECS:
        raise SkipStage(
            f"Clip too short ({duration:.1f}s < {AUDIO_MIN_DURATION_SECS}s) — skipping audio context"
        )
    if duration > AUDIO_MAX_DURATION_SECS:
        raise SkipStage(
            f"Clip too long ({duration:.0f}s > {AUDIO_MAX_DURATION_SECS:.0f}s) — "
            f"skipping audio context to control cost"
        )

    input_video: Optional[Path] = None
    if ctx.processed_video_path and ctx.processed_video_path.exists():
        input_video = ctx.processed_video_path
    elif ctx.local_video_path and ctx.local_video_path.exists():
        input_video = ctx.local_video_path

    if not input_video:
        raise SkipStage("No video file available for audio extraction")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available for audio extraction")

    logger.info(
        f"Audio context: {input_video.name} | duration={duration:.1f}s | codec={audio_codec} | "
        f"whisper={want_whisper} yamnet={want_yamnet} acr={want_acr} gpt_sum={want_gpt_summary}"
    )

    ctx.audio_context = dict(getattr(ctx, "audio_context", None) or {})

    wav_path = ctx.temp_dir / "audio_context.wav"
    extracted = await _extract_audio_wav(input_video, wav_path)
    if not extracted:
        logger.warning("Audio extraction failed — skipping audio sub-services")
        return ctx

    ctx.audio_path = wav_path

    if want_whisper and OPENAI_API_KEY:
        transcript = await _transcribe_wav(wav_path)
        if transcript:
            if len(transcript) > TRANSCRIPT_MAX_CHARS:
                transcript = transcript[:TRANSCRIPT_MAX_CHARS] + "…"
            ctx.ai_transcript = transcript
            ctx.audio_context["transcript_chars"] = len(transcript)
            logger.info("Whisper transcript stored (%s chars)", len(transcript))
    elif want_whisper:
        logger.warning("Speech-to-text requested but OPENAI_API_KEY is not set")

    if want_yamnet:
        ctx.audio_context.setdefault("yamnet_events", [])
        logger.debug("YAMNet: no local classifier loaded — pref honored, events left empty")

    if want_acr:
        ctx.audio_context.setdefault("music_fingerprint", None)
        logger.debug("ACRCloud: client not wired in worker — pref honored, skipped")

    if want_gpt_summary and OPENAI_API_KEY:
        base = (ctx.ai_transcript or "").strip()
        if base:
            summary = await _gpt_audio_summary_from_transcript(base)
            if summary:
                ctx.audio_context["gpt_audio_summary"] = summary
    elif want_gpt_summary:
        logger.debug("Audio summary requested but no OpenAI key or empty transcript")

    return ctx
