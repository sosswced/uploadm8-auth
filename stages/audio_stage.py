"""
UploadM8 Audio Context Stage
==============================
Extract audio from video, then (per user prefs + env):

- **OpenAI Whisper** — speech-to-text → ``ctx.ai_transcript``
- **OpenAI GPT** — short prose summary of the transcript
- **ACRCloud** — music catalogue identification (needs ``ACRCLOUD_HOST``, key, secret)
- **YAMNet** (TensorFlow Hub) — environmental / AudioSet sound tags (optional deps;
  see ``requirements-audio-ml.txt``)

Flow:
  1. Check entitlements + user setting (use_audio_context)
  2. Skip if no audio stream in video_info
  3. Extract 16 kHz mono WAV using FFmpeg (-vn)
  4. Whisper (optional) → ``ctx.ai_transcript`` (capped)
  5. Short head clip (~15s) for ACR + YAMNet when those services are on
  6. ACRCloud identify (optional) → ``music_*`` fields on ``ctx.audio_context``
  7. YAMNet (optional) → ``yamnet_events``, ``sound_profile``, ``top_sound_class``
  8. GPT audio summary (optional)

Placement: After transcode_stage, BEFORE thumbnail_stage AND caption_stage.

Graceful: Any failure at any sub-step → log warning + continue; captions may
          miss that signal only.

Exports: run_audio_context_stage(ctx)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from .errors import SkipStage
from .context import JobContext
from .ai_service_costs import billing_env_from_os, user_pref_ai_service_enabled


def _wav_speech_like_energy(wav_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap RMS / peak check on extracted WAV to detect speech-like energy.
    Used to auto-enable Whisper when prefs left STT off but the clip talks.
    """
    meta: Dict[str, Any] = {"ok": False}
    try:
        import audioop
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            nframes = wf.getnframes()
            # Sample up to ~8s from the middle for speed.
            take = min(nframes, int(rate * 8))
            start = max(0, (nframes - take) // 2)
            wf.setpos(start)
            raw = wf.readframes(take)
        if not raw:
            return False, meta
        rms = audioop.rms(raw, sw)
        peak = audioop.max(raw, sw)
        # 16-bit PCM: silence ~0–200; speech often >400–800 RMS.
        thresh = int(os.environ.get("MULTIMODAL_SPEECH_RMS_MIN", "450") or 450)
        speech_like = rms >= thresh and peak >= thresh * 2
        meta.update(
            {
                "ok": True,
                "rms": int(rms),
                "peak": int(peak),
                "rate": rate,
                "channels": nch,
                "thresh": thresh,
                "speech_like": speech_like,
            }
        )
        return speech_like, meta
    except Exception as e:
        meta["error"] = str(e)[:160]
        return False, meta
from .ffmpeg_env import resolve_ffmpeg_executable

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

# Set AUDIO_STAGE_ENABLED=false to disable the entire audio analysis stack at
# the infrastructure level (e.g. on workers without audio libraries or on
# video-only ingestion pipelines).
AUDIO_STAGE_ENABLED = os.environ.get("AUDIO_STAGE_ENABLED", "true").lower() not in (
    "0", "false", "no", "off"
)


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
    _ffmpeg = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        _ffmpeg, "-y",
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


async def _ffmpeg_slice_wav_head(src: Path, dst: Path, seconds: float) -> bool:
    """Write the first ``seconds`` of a WAV/PCM file to ``dst`` (16 kHz mono PCM)."""
    if seconds <= 0 or not src.is_file():
        return False
    _ffmpeg = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        _ffmpeg,
        "-y",
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
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
            logger.warning(
                "FFmpeg head slice failed rc=%s: %s",
                proc.returncode,
                stderr_bytes.decode(errors="replace")[-500:],
            )
            return False
        if not dst.exists() or dst.stat().st_size < 200:
            return False
        return True
    except asyncio.TimeoutError:
        logger.warning("FFmpeg head slice timed out after %ss", FFMPEG_AUDIO_TIMEOUT)
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except Exception as exc:
        logger.warning("FFmpeg head slice error: %s", exc)
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


async def _transcribe_wav(wav_path: Path) -> Optional[Dict[str, Any]]:
    """
    Send a WAV file to OpenAI Whisper and return a structured transcript payload.

    Uses ``response_format=verbose_json`` so the API returns segments
    (with start/end timestamps), detected language, and overall duration.
    These are the building blocks the M8 engine and hydration enforcer
    use for "speech-anchored" caption variants and per-topic timestamps.

    Returns a dict with at least ``text``; may also contain ``language``,
    ``duration``, and ``segments`` — or ``None`` on any failure.
    """
    if not OPENAI_API_KEY:
        logger.debug("OPENAI_API_KEY not set — cannot call Whisper")
        return None

    try:
        import httpx

        file_size_kb = wav_path.stat().st_size / 1024
        logger.info(
            f"Sending {file_size_kb:.1f} KB WAV to Whisper API (model={WHISPER_MODEL}, verbose_json)"
        )

        async with httpx.AsyncClient(timeout=120) as client:
            with open(wav_path, "rb") as audio_file:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    data={
                        "model": WHISPER_MODEL,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment",
                    },
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
        text = (payload.get("text") or "").strip()

        if not text:
            logger.info("Whisper returned an empty transcript")
            return None

        seg_in = payload.get("segments") or []
        segments: list = []
        for s in seg_in:
            if not isinstance(s, dict):
                continue
            seg_text = str(s.get("text") or "").strip()
            if not seg_text:
                continue
            try:
                start = float(s.get("start") or 0.0)
                end = float(s.get("end") or 0.0)
            except (TypeError, ValueError):
                start = 0.0
                end = 0.0
            segments.append(
                {
                    "id": s.get("id"),
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": seg_text,
                    "no_speech_prob": s.get("no_speech_prob"),
                    "avg_logprob": s.get("avg_logprob"),
                }
            )

        duration = payload.get("duration")
        try:
            duration_f = float(duration) if duration is not None else 0.0
        except (TypeError, ValueError):
            duration_f = 0.0

        logger.info(
            f"Whisper transcript received ({len(text)} chars, {len(segments)} segments, "
            f"lang={payload.get('language')!r}): {text[:120]}{'...' if len(text) > 120 else ''}"
        )
        return {
            "text": text,
            "language": payload.get("language") or "",
            "duration": duration_f,
            "segments": segments,
        }

    except Exception as exc:
        logger.warning(f"Whisper transcription failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Whisper STRUCTURED post-processing — entities/topics/questions/speaker turns
# ---------------------------------------------------------------------------

# Heuristic regexes are deliberately cheap so we don't add latency for clips
# that have only one or two short utterances; the GPT pass below adds the
# higher-quality NER / question / topic extraction when an OPENAI_API_KEY
# is configured. Both layers feed into ``transcript_structured`` so the
# downstream M8 prompt and hydration enforcer always have *some* signal.

import re  # noqa: E402  (module-level import deferred for grouping)
from typing import Any, Dict  # noqa: E402

_QUESTION_RE = re.compile(r"([A-Z][^.!?]{4,200}\?)")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_PRICE_RE = re.compile(r"(\$\d{1,3}(?:[,]\d{3})*(?:\.\d{1,2})?)")
_NUMBERS_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(mph|miles?|kilometers?|km|hours?|minutes?|secs?|seconds?|years?|days?|weeks?|months?)\b", re.I)
_HASHTAG_KEYWORD_STOP = {
    "the", "and", "but", "for", "with", "this", "that", "they", "them",
    "from", "into", "your", "yours", "have", "had", "has", "are", "was",
    "were", "you", "i'm", "i'll", "i've", "it's", "we're", "they're",
}


def _heuristic_structured_extract(text: str, segments: list) -> Dict[str, Any]:
    """Cheap regex-based pass: questions, proper nouns, key noun phrases."""
    blob = (text or "").strip()
    if not blob:
        return {}
    questions = list({q.strip() for q in _QUESTION_RE.findall(blob)})[:8]
    propers = list({p.strip() for p in _PROPER_NOUN_RE.findall(blob) if len(p) >= 4})[:24]
    prices = list({p.strip() for p in _PRICE_RE.findall(blob)})[:6]
    measures = [(num, unit) for num, unit in _NUMBERS_RE.findall(blob)][:8]

    # Keyword scoring: word frequency minus stoplist (poor man's TF)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", blob.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w in _HASHTAG_KEYWORD_STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    top_keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:14]]

    # Per-topic timestamps from segments: bucket segments by their dominant keyword
    topic_timestamps: Dict[str, Dict[str, Any]] = {}
    for seg in segments or []:
        seg_text = str(seg.get("text") or "").lower()
        if not seg_text:
            continue
        for kw in top_keywords[:6]:
            if kw and kw in seg_text:
                bucket = topic_timestamps.setdefault(
                    kw, {"first_seen": seg.get("start"), "last_seen": seg.get("end"), "mentions": 0}
                )
                bucket["mentions"] = int(bucket.get("mentions") or 0) + 1
                bucket["last_seen"] = seg.get("end")
                break

    return {
        "engine": "heuristic",
        "questions": questions,
        "proper_nouns": propers,
        "prices": prices,
        "measurements": [{"value": v, "unit": u} for v, u in measures],
        "top_keywords": top_keywords,
        "topic_timestamps": topic_timestamps,
        "key_phrase": (questions[0] if questions else (blob.split(".")[0].strip()[:140] or "")),
    }


_GPT_STRUCTURED_PROMPT = (
    "Extract structured signals from this video transcript. Return STRICT JSON only, no prose.\n"
    "Schema:\n"
    "{\n"
    '  "language": "ISO-639 code (en, es, etc.)",\n'
    '  "named_entities": {"people": [], "places": [], "products": [], "organizations": []},\n'
    '  "topics": [3-8 short noun phrases describing what is discussed],\n'
    '  "questions": [questions actually asked, verbatim],\n'
    '  "key_phrase": "the single best 6-12 word quote or paraphrase that captures the hook",\n'
    '  "speaker_turns": [{"speaker": "A", "summary": "what speaker A is talking about"}],\n'
    '  "sentiment": "positive|neutral|negative|mixed",\n'
    '  "topic_timestamps": [{"topic": "...", "start_seconds": 0.0, "end_seconds": 0.0}]\n'
    "}\n"
    "Use only what's in the transcript. If a field has no evidence return [] or null.\n"
    "Speaker labels: only assign multiple speakers when speech style/voice obviously changes.\n"
    "topic_timestamps: derive from segment boundaries where each topic dominates.\n"
)


async def _gpt_structured_transcript(
    text: str, segments: list, language_hint: str = ""
) -> Optional[Dict[str, Any]]:
    """GPT-4o-mini structured pass: entities, topics, questions, speaker turns."""
    if not text or not OPENAI_API_KEY:
        return None
    model = os.environ.get("OPENAI_TRANSCRIPT_STRUCTURED_MODEL", "gpt-4o-mini")
    seg_lines = []
    for s in (segments or [])[:80]:
        try:
            start = float(s.get("start") or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        seg_lines.append(f"[{start:6.1f}s] {str(s.get('text') or '').strip()}")
    seg_block = "\n".join(seg_lines)[:6000]
    prompt = (
        _GPT_STRUCTURED_PROMPT
        + (f"\nDetected language hint: {language_hint}\n" if language_hint else "")
        + "\nTranscript (full text):\n"
        + (text[:5000])
        + "\n\nSegment timeline (start → text):\n"
        + seg_block
    )
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 700,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            if r.status_code != 200:
                logger.warning("Whisper structured GPT HTTP %s: %s", r.status_code, r.text[:200])
                return None
            data = r.json()
            raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            import json as _json
            try:
                parsed = _json.loads(raw)
            except Exception as e:
                logger.warning("Whisper structured GPT JSON parse failed: %s", e)
                return None
            if not isinstance(parsed, dict):
                return None
            parsed["engine"] = "openai_gpt"
            return parsed
    except Exception as e:
        logger.warning("Whisper structured GPT call failed: %s", e)
        return None


async def _build_structured_transcript(
    text: str, segments: list, language: str
) -> Dict[str, Any]:
    """Always returns a non-empty dict when text exists.

    Heuristic pass runs locally (free, fast). If OPENAI is configured, GPT
    pass merges higher-quality entities/topics on top. The merged result is
    written to ``audio_context.transcript_structured`` and consumed by both
    the M8 prompt and the hydration enforcer.
    """
    base = _heuristic_structured_extract(text, segments)
    gpt = await _gpt_structured_transcript(text, segments, language)
    if not gpt:
        return base or {}
    merged: Dict[str, Any] = dict(base or {})
    merged["engine"] = "openai_gpt+heuristic"
    if gpt.get("language"):
        merged["language"] = gpt["language"]
    ne = gpt.get("named_entities") or {}
    if isinstance(ne, dict):
        merged["named_entities"] = {
            k: [str(x).strip() for x in (ne.get(k) or []) if str(x).strip()][:12]
            for k in ("people", "places", "products", "organizations")
        }
    for k in ("topics", "questions", "speaker_turns"):
        v = gpt.get(k) or []
        if isinstance(v, list):
            merged[k] = v[:12] if k != "speaker_turns" else v[:6]
    if gpt.get("key_phrase"):
        merged["key_phrase"] = str(gpt["key_phrase"])[:200]
    if gpt.get("sentiment"):
        merged["sentiment"] = str(gpt["sentiment"])[:20]
    tt_gpt = gpt.get("topic_timestamps") or []
    if isinstance(tt_gpt, list) and tt_gpt:
        merged["topic_timestamps_ai"] = tt_gpt[:12]
    return merged


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

async def run_audio_context_stage(ctx: JobContext) -> JobContext:
    """
    Audio analysis stack: Whisper, GPT summary, ACRCloud music ID, YAMNet env tags.

    Respects per-service upload preferences (aiService* + audio_transcription).
    """
    ctx.mark_stage("audio")

    if not AUDIO_STAGE_ENABLED:
        raise SkipStage("Audio stage disabled by AUDIO_STAGE_ENABLED=false")

    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI not enabled for this tier — skipping audio context")

    us = ctx.user_settings or {}
    use_audio = bool(us.get("use_audio_context", us.get("useAudioContext", True)))
    if not use_audio:
        raise SkipStage("Audio context disabled by user setting (use_audio_context=false)")

    env_audio = billing_env_from_os()
    transcribe_pref = bool(us.get("audio_transcription", us.get("audioTranscription", True)))
    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    want_whisper = transcribe_pref and user_pref_ai_service_enabled(
        us, "audio_whisper", default=False, allowed_services=tier_allowed_set
    )
    want_yamnet = env_audio.get("YAMNET_ENABLED", True) and user_pref_ai_service_enabled(
        us, "audio_yamnet", default=True, allowed_services=tier_allowed_set
    )
    want_acr = env_audio.get("ACRCLOUD_CONFIGURED", False) and user_pref_ai_service_enabled(
        us, "audio_acr", default=True, allowed_services=tier_allowed_set
    )
    want_gpt_summary = user_pref_ai_service_enabled(
        us, "audio_gpt_classify", default=True, allowed_services=tier_allowed_set
    )

    auto_whisper_env = (os.environ.get("MULTIMODAL_AUTO_WHISPER_ON_SPEECH") or "true").strip().lower()
    auto_whisper_on = auto_whisper_env in ("1", "true", "yes", "on")
    # Explicit user off (aiServiceSpeechToText=false) still wins; only auto-enable
    # when the pref was left at default-off / unset and use_audio is on.
    explicit_stt_off = (
        us.get("aiServiceSpeechToText") is False
        or us.get("ai_service_speech_to_text") is False
        or us.get("audio_transcription") is False
        or us.get("audioTranscription") is False
    )

    if not any((want_whisper, want_yamnet, want_acr, want_gpt_summary, auto_whisper_on and not explicit_stt_off)):
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

    if want_whisper and not OPENAI_API_KEY:
        logger.warning(
            "Speech-to-text is enabled in user preferences (aiServiceSpeechToText) but "
            "OPENAI_API_KEY is not set — Whisper will be skipped and captions will lack "
            "spoken-word evidence."
        )

    ctx.audio_context = dict(getattr(ctx, "audio_context", None) or {})

    wav_path = ctx.temp_dir / "audio_context.wav"
    extracted = await _extract_audio_wav(input_video, wav_path)
    if not extracted:
        logger.warning("Audio extraction failed — skipping audio sub-services")
        return ctx

    ctx.audio_path = wav_path

    # Minimum viable watch: if STT wasn't requested but the WAV has speech-like
    # energy, auto-enable Whisper so M8 gets spoken-word evidence.
    speech_meta: Dict[str, Any] = {}
    if (
        not want_whisper
        and auto_whisper_on
        and not explicit_stt_off
        and OPENAI_API_KEY
        and (tier_allowed_set is None or "audio_whisper" in tier_allowed_set)
    ):
        speech_like, speech_meta = _wav_speech_like_energy(wav_path)
        ctx.audio_context["speech_energy"] = speech_meta
        if speech_like:
            want_whisper = True
            ctx.audio_context["whisper_auto_enabled"] = True
            logger.info(
                "Auto-enabled Whisper (speech-like energy rms=%s peak=%s)",
                speech_meta.get("rms"),
                speech_meta.get("peak"),
            )
    elif not want_whisper:
        speech_like, speech_meta = _wav_speech_like_energy(wav_path)
        ctx.audio_context["speech_energy"] = speech_meta
        ctx.audio_context["speech_like_hint"] = bool(speech_like)

    if want_whisper and OPENAI_API_KEY:
        whisper_payload = await _transcribe_wav(wav_path)
        if whisper_payload:
            transcript = whisper_payload.get("text") or ""
            language = whisper_payload.get("language") or ""
            duration_s = float(whisper_payload.get("duration") or 0.0)
            segments = whisper_payload.get("segments") or []
            if len(transcript) > TRANSCRIPT_MAX_CHARS:
                transcript = transcript[:TRANSCRIPT_MAX_CHARS] + "…"
            ctx.ai_transcript = transcript
            ctx.audio_context["transcript_chars"] = len(transcript)
            ctx.audio_context["transcript_language"] = language
            ctx.audio_context["language"] = ctx.audio_context.get("language") or language
            ctx.audio_context["transcript_duration"] = duration_s
            ctx.audio_context["transcript_segments"] = segments[:120]
            logger.info(
                "Whisper transcript stored (%s chars, %s segments, lang=%s)",
                len(transcript),
                len(segments),
                language or "?",
            )
            # Structured pass: always run (heuristic free; GPT optional via env).
            try:
                structured = await _build_structured_transcript(transcript, segments, language)
                if structured:
                    ctx.audio_context["transcript_structured"] = structured
                    nq = len(structured.get("questions") or [])
                    nt = len(structured.get("topics") or [])
                    ne = structured.get("named_entities") or {}
                    np_ = sum(len(ne.get(k) or []) for k in ("people", "places", "products", "organizations")) if isinstance(ne, dict) else 0
                    logger.info(
                        "Whisper structured: engine=%s questions=%s topics=%s entities=%s",
                        structured.get("engine"),
                        nq,
                        nt,
                        np_,
                    )
            except Exception as e:
                logger.warning("Whisper structured extraction failed: %s", e)
    elif want_whisper:
        logger.warning("Speech-to-text requested but OPENAI_API_KEY is not set")

    if ctx.ai_transcript:
        ctx.audio_context.setdefault("transcript", (ctx.ai_transcript or "")[:7500])

    clip_path = ctx.temp_dir / "audio_ml_clip.wav"
    clip_ok = False
    if want_acr or want_yamnet:
        clip_sec = min(15.0, max(1.0, duration))
        clip_ok = await _ffmpeg_slice_wav_head(wav_path, clip_path, clip_sec)
        if not clip_ok:
            logger.warning("Audio ML: could not build head clip — ACR may be skipped for huge files")

    ml_wav = clip_path if clip_ok else wav_path

    if want_acr and clip_ok:
        try:
            from services.acrcloud_identify import identify_music_file_sync

            acr_meta = await asyncio.to_thread(identify_music_file_sync, ml_wav)
            if acr_meta:
                for k, v in acr_meta.items():
                    if v is not None and v != "":
                        ctx.audio_context[k] = v
                if acr_meta.get("music_detected"):
                    cs = [str(x) for x in (ctx.audio_context.get("content_signals") or []) if x]
                    if "acr_catalog_match" not in cs:
                        cs.append("acr_catalog_match")
                    ctx.audio_context["content_signals"] = cs[:32]
        except Exception as e:
            logger.warning("ACRCloud identify error: %s", e)
    elif want_acr and not clip_ok:
        logger.warning("ACRCloud: skipped (no head clip — configure FFmpeg or shorten source)")

    if want_yamnet:
        try:
            from services.yamnet_env import classify_wav_path

            yn = await asyncio.to_thread(classify_wav_path, ml_wav)
            if yn.get("yamnet_events"):
                ctx.audio_context["yamnet_events"] = yn["yamnet_events"]
            if yn.get("top_sound_class"):
                ctx.audio_context["top_sound_class"] = yn["top_sound_class"]
            if yn.get("sound_profile"):
                ctx.audio_context["sound_profile"] = yn["sound_profile"]
            if yn.get("yamnet_scores"):
                ctx.audio_context["yamnet_scoreboard"] = yn["yamnet_scores"]
        except Exception as e:
            logger.warning("YAMNet classify error: %s", e)

    kw = [str(x).strip() for x in (ctx.audio_context.get("suggested_keywords") or []) if str(x).strip()]
    for slug in ctx.audio_context.get("yamnet_events") or []:
        s = str(slug).strip().lower()
        if s and s not in kw:
            kw.append(s[:48])
    for bit in (
        str(ctx.audio_context.get("music_genre") or "").strip(),
        str(ctx.audio_context.get("music_title") or "").strip(),
        str(ctx.audio_context.get("music_artist") or "").strip(),
    ):
        b = bit.lower().replace(" ", "")
        if len(b) >= 3 and b not in kw:
            kw.append(b[:48])
    if kw:
        ctx.audio_context["suggested_keywords"] = kw[:48]

    if not str(ctx.audio_context.get("fusion_narrative") or "").strip():
        fusion_bits = []
        mt = str(ctx.audio_context.get("music_title") or "").strip()
        ma = str(ctx.audio_context.get("music_artist") or "").strip()
        if mt or ma:
            fusion_bits.append(
                ("Catalog music: " + (f"{ma} — " if ma else "") + mt).strip()
            )
        spf = str(ctx.audio_context.get("sound_profile") or "").strip()
        if spf:
            fusion_bits.append(spf[:900])
        if fusion_bits:
            ctx.audio_context["fusion_narrative"] = " | ".join(fusion_bits)[:1900]

    if want_gpt_summary and OPENAI_API_KEY:
        base = (ctx.ai_transcript or "").strip()
        if base:
            summary = await _gpt_audio_summary_from_transcript(base)
            if summary:
                ctx.audio_context["gpt_audio_summary"] = summary
    elif want_gpt_summary:
        logger.debug("Audio summary requested but no OpenAI key or empty transcript")

    return ctx
