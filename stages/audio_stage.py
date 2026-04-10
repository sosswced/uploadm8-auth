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

from .errors import SkipStage, StageError, ErrorCode
from .context import JobContext

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
    Execute the audio context stage.

    On success:   ctx.ai_transcript is set (str, ≤ TRANSCRIPT_MAX_CHARS chars).
                  ctx.audio_path is set to the temporary WAV path.
    On any skip:  SkipStage is raised — pipeline continues unchanged.
    On any error: logs a warning, returns ctx unchanged (never raises to caller).

    The caption_stage reads ctx.ai_transcript in _build_grounding_evidence()
    and injects it into every OpenAI prompt as factual evidence, giving the
    model "what was actually said" in addition to "what was actually shown."
    """
    ctx.mark_stage("audio")

    # ------------------------------------------------------------------ #
    # Gate 1: Tier must have AI enabled                                   #
    # ------------------------------------------------------------------ #
    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI not enabled for this tier — skipping audio context")

    if not OPENAI_API_KEY:
        raise SkipStage("OPENAI_API_KEY not configured — skipping audio context")

    # ------------------------------------------------------------------ #
    # Gate 2: User opt-out via settings                                   #
    # Default True so existing users get the feature automatically.       #
    # ------------------------------------------------------------------ #
    use_audio = ctx.user_settings.get("use_audio_context", True)
    if not use_audio:
        raise SkipStage("Audio context disabled by user setting (use_audio_context=false)")

    # ------------------------------------------------------------------ #
    # Gate 3: Video must have an audio stream                             #
    # video_info is populated by transcode_stage from ffprobe output.     #
    # ------------------------------------------------------------------ #
    audio_codec = ctx.video_info.get("audio_codec")
    if not audio_codec:
        raise SkipStage("Video has no audio stream — skipping audio context")

    # ------------------------------------------------------------------ #
    # Gate 4: Duration bounds (cost + quality guards)                     #
    # ------------------------------------------------------------------ #
    duration = float(ctx.video_info.get("duration", 0.0))
    if duration < AUDIO_MIN_DURATION_SECS:
        raise SkipStage(
            f"Clip too short ({duration:.1f}s < {AUDIO_MIN_DURATION_SECS}s) — skipping audio context"
        )
    if duration > AUDIO_MAX_DURATION_SECS:
        raise SkipStage(
            f"Clip too long ({duration:.0f}s > {AUDIO_MAX_DURATION_SECS:.0f}s) — "
            f"skipping audio context to control Whisper cost"
        )

    # ------------------------------------------------------------------ #
    # Resolve input video                                                  #
    # Prefer the processed (HUD+watermarked+transcoded) path.             #
    # The audio stream in the processed file is identical to the original #
    # (FFmpeg copies it through losslessly in most cases).                #
    # ------------------------------------------------------------------ #
    input_video: Optional[Path] = None
    if ctx.processed_video_path and ctx.processed_video_path.exists():
        input_video = ctx.processed_video_path
    elif ctx.local_video_path and ctx.local_video_path.exists():
        input_video = ctx.local_video_path

    if not input_video:
        raise SkipStage("No video file available for audio extraction")

    logger.info(
        f"Audio context: {input_video.name} | "
        f"duration={duration:.1f}s | codec={audio_codec}"
    )

    # ------------------------------------------------------------------ #
    # Step 1: Extract audio → WAV                                         #
    # ------------------------------------------------------------------ #
    wav_path = ctx.temp_dir / "audio_context.wav"
    extracted = await _extract_audio_wav(input_video, wav_path)

    if not extracted:
        # Non-fatal: caption stage will run without transcript
        logger.warning("Audio extraction failed — continuing without transcript")
        return ctx

    ctx.audio_path = wav_path  # stored for reference; temp_dir cleans up automatically

    # ------------------------------------------------------------------ #
    # Step 2: Transcribe with Whisper                                     #
    # ------------------------------------------------------------------ #
    transcript = await _transcribe_wav(wav_path)

    if not transcript:
        logger.info("No transcript produced — continuing without audio context")
        return ctx

    # ------------------------------------------------------------------ #
    # Step 3: Cap length and store in context                             #
    # ------------------------------------------------------------------ #
    if len(transcript) > TRANSCRIPT_MAX_CHARS:
        transcript = transcript[:TRANSCRIPT_MAX_CHARS] + "…"
        logger.info(f"Transcript capped at {TRANSCRIPT_MAX_CHARS} chars")

    ctx.ai_transcript = transcript
    logger.info(
        f"Audio context stage complete — "
        f"transcript stored ({len(ctx.ai_transcript)} chars)"
    )

    return ctx
