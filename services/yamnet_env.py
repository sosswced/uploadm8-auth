"""
Local environmental sound tags via YAMNet (TensorFlow Hub).

Optional dependency: ``tensorflow`` + ``tensorflow-hub`` (see requirements-audio-ml.txt).
If imports fail, helpers return empty structures — the worker still runs with
Whisper + ACRCloud only.

Input: mono 16-bit PCM WAV at 16 kHz (same as audio_stage extraction).
"""

from __future__ import annotations

import logging
import os
import struct
import wave
import csv
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("uploadm8-worker.yamnet")

YAMNET_MAX_SECONDS = float(os.environ.get("YAMNET_MAX_AUDIO_SECONDS", "15.0"))
YAMNET_TOP_K = int(os.environ.get("YAMNET_TOP_K", "8"))

# Log once when YAMNet cannot run so operators know the worker is healthy without TF.
_yamnet_skip_notice_logged = False


def _read_wav_mono_float32(path: Path) -> Tuple[Any, int]:
    """Return (waveform float32 tensor/array-like, sample_rate). Empty on failure."""
    try:
        with wave.open(str(path), "rb") as w:
            nch = w.getnchannels()
            sw = w.getsampwidth()
            fr = w.getframerate()
            nframes = w.getnframes()
            if fr != 16000:
                logger.warning("YAMNet: expected 16 kHz WAV, got %s — skipping", fr)
                return None, fr
            max_frames = int(YAMNET_MAX_SECONDS * fr)
            to_read = min(nframes, max_frames)
            raw = w.readframes(to_read)
    except (wave.Error, OSError, struct.error) as e:
        logger.warning("YAMNet: could not read WAV %s: %s", path, e)
        return None, 0

    if sw != 2 or nch not in (1, 2):
        logger.warning("YAMNet: need 16-bit PCM mono or stereo WAV")
        return None, 0

    n_samples = len(raw) // (sw * nch)
    import numpy as np

    flat = np.frombuffer(raw, dtype=np.int16)
    if nch == 2:
        flat = flat.reshape(-1, 2).mean(axis=1).astype(np.int16)
    waveform = flat.astype("float32") / 32768.0
    return waveform, 16000


@lru_cache(maxsize=1)
def _yamnet_bundle():
    """Load TF Hub YAMNet once per process."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        logger.info("YAMNet: tensorflow not installed (%s) — audio event tags disabled", e)
        return None

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        return model
    except Exception as e:
        logger.warning("YAMNet: failed to load hub model: %s", e)
        return None


@lru_cache(maxsize=1)
def _class_names() -> List[str]:
    """521 AudioSet class display names aligned with YAMNet score vector index."""
    local_path = (os.environ.get("YAMNET_CLASS_MAP_PATH") or "").strip()
    text = ""

    if local_path:
        try:
            with open(local_path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            logger.warning("YAMNet: could not read YAMNET_CLASS_MAP_PATH %s: %s", local_path, e)

    if not text:
        try:
            import urllib.request

            url = (
                "https://raw.githubusercontent.com/tensorflow/models/"
                "master/research/audioset/yamnet/yamnet_class_map.csv"
            )
            with urllib.request.urlopen(url, timeout=20) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(
                "YAMNet: could not download class map (%s). "
                "Set YAMNET_CLASS_MAP_PATH to a local copy to avoid network dependency.",
                e,
            )
            return []

    by_index: Dict[int, str] = {}
    for row in csv.reader(StringIO(text)):
        if len(row) < 3:
            continue
        try:
            idx = int(row[0].strip())
        except ValueError:
            continue
        by_index[idx] = row[2].strip().strip('"')
    if not by_index:
        return []
    max_i = max(by_index.keys())
    out = [by_index.get(i, f"class_{i}") for i in range(max_i + 1)]
    return out


def _empty_yamnet_result() -> Dict[str, Any]:
    """Stable shape when TensorFlow is missing, model load fails, or inference errors."""
    return {
        "yamnet_events": [],
        "top_sound_class": "",
        "sound_profile": "",
        "yamnet_scores": [],
    }


def classify_wav_path(path: Path) -> Dict[str, Any]:
    """
    Run YAMNet on a WAV file. Returns dict with:

    - yamnet_events: list[str] (slug-like tokens for hashtags / scene tags)
    - top_sound_class: str
    - sound_profile: short human summary
    - yamnet_scores: optional list of {name, score} for debugging

    **Never raises.** If ``tensorflow`` / ``tensorflow-hub`` are not installed, or the
    model fails to load or run, returns the same empty structure and the worker
    continues (Whisper, GPT audio summary, ACRCloud still apply).
    """
    empty = _empty_yamnet_result()
    global _yamnet_skip_notice_logged

    try:
        model = _yamnet_bundle()
        if model is None:
            if not _yamnet_skip_notice_logged:
                _yamnet_skip_notice_logged = True
                logger.info(
                    "YAMNet: disabled (TensorFlow Hub model unavailable — install "
                    "requirements-audio-ml.txt on workers for environmental tags). "
                    "Pipeline continues without yamnet_events."
                )
            return empty

        waveform, sr = _read_wav_mono_float32(path)
        if waveform is None or sr != 16000:
            return empty

        import numpy as np
        import tensorflow as tf

        wav = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, _, _ = model(wav)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_k = max(1, min(YAMNET_TOP_K, 32))
        idx = np.argsort(-mean_scores)[:top_k]
        cmap = _class_names()
        events: List[str] = []
        score_rows: List[Dict[str, Any]] = []
        for i in idx:
            score = float(mean_scores[int(i)])
            if score < 0.08:
                continue
            label = cmap[int(i)] if cmap and int(i) < len(cmap) else f"class_{int(i)}"
            slug = (
                label.lower()
                .replace(" ", "")
                .replace("'", "")
                .replace("/", "")
                .replace(",", "")[:48]
            )
            if slug and slug not in events:
                events.append(slug)
            score_rows.append({"name": label, "score": round(score, 4)})
        top_sound = events[0] if events else ""
        sound_profile = (
            "Dominant environmental sounds (AudioSet/YAMNet): " + ", ".join(events[:6])
            if events
            else ""
        )
        return {
            "yamnet_events": events[:24],
            "top_sound_class": top_sound,
            "sound_profile": sound_profile[:500],
            "yamnet_scores": score_rows[:12],
        }
    except Exception as e:
        logger.warning("YAMNet: classify_wav_path failed (returning empty): %s", e)
        return empty
