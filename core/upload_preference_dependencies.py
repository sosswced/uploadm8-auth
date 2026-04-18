"""
Keep upload / AI preference toggles consistent (parent → child).

- When audio context is off, all audio sub-services and speech-to-text are off.
- When auto-thumbnails is off, styled + Thumbnail Studio toggles are off.
- When Thumbnail Studio master is off, engine + persona are off.
- When Drive analysis (trill) is off, telemetry insights service is off.
- When caption writer is off, frame/scene/video analyzer services are off (they feed captions).

Used on ``users.preferences`` merge (camelCase) and on snake_case payloads for ``user_preferences``.
"""


from __future__ import annotations

from typing import Any, Dict, Mapping


def _b(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, str):
        return val.lower() not in ("false", "0", "no", "off", "")
    return bool(val)


def _get(d: Mapping[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    if snake in d and d[snake] is not None:
        return d[snake]
    if camel in d and d[camel] is not None:
        return d[camel]
    return default


def _set_pair(d: Dict[str, Any], snake: str, camel: str, val: Any) -> None:
    d[snake] = val
    d[camel] = val


def normalize_preferences_dict(d: Dict[str, Any]) -> None:
    """
    Mutate a preferences dict in place. Supports snake_case, camelCase, or both.
    """
    # Third-party voice-emotion (Hume) integration removed — preference stays off.
    _set_pair(d, "ai_service_emotion_signals", "aiServiceEmotionSignals", False)

    use_audio = _b(_get(d, "use_audio_context", "useAudioContext"), True)
    if not use_audio:
        _set_pair(d, "use_audio_context", "useAudioContext", False)
        _set_pair(d, "audio_transcription", "audioTranscription", False)
        _set_pair(d, "ai_service_speech_to_text", "aiServiceSpeechToText", False)
        _set_pair(d, "ai_service_audio_signals", "aiServiceAudioSignals", False)
        _set_pair(d, "ai_service_music_detection", "aiServiceMusicDetection", False)
        _set_pair(d, "ai_service_audio_summary", "aiServiceAudioSummary", False)
        _set_pair(d, "ai_service_emotion_signals", "aiServiceEmotionSignals", False)

    if not _b(_get(d, "audio_transcription", "audioTranscription"), True):
        _set_pair(d, "ai_service_speech_to_text", "aiServiceSpeechToText", False)

    auto_thumb = _b(_get(d, "auto_thumbnails", "autoThumbnails"), False)
    if not auto_thumb:
        _set_pair(d, "styled_thumbnails", "styledThumbnails", False)
        _set_pair(d, "thumbnail_studio_enabled", "thumbnailStudioEnabled", False)
        _set_pair(d, "thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled", False)
        _set_pair(d, "thumbnail_persona_enabled", "thumbnailPersonaEnabled", False)

    studio = _b(_get(d, "thumbnail_studio_enabled", "thumbnailStudioEnabled"), False)
    if not studio:
        _set_pair(d, "thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled", False)
        _set_pair(d, "thumbnail_persona_enabled", "thumbnailPersonaEnabled", False)

    trill = _b(_get(d, "trill_enabled", "trillEnabled"), False)
    if not trill:
        _set_pair(d, "ai_service_telemetry", "aiServiceTelemetry", False)

    cap = _b(_get(d, "ai_service_caption_writer", "aiServiceCaptionWriter"), True)
    if not cap:
        _set_pair(d, "ai_service_frame_inspector", "aiServiceFrameInspector", False)
        _set_pair(d, "ai_service_scene_understanding", "aiServiceSceneUnderstanding", False)
        _set_pair(d, "ai_service_video_analyzer", "aiServiceVideoAnalyzer", False)


def normalize_upload_preferences_snake(p: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy normalized; snake_case keys (for user_preferences row)."""
    d = dict(p)
    normalize_preferences_dict(d)
    return d
