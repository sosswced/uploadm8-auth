"""
Keep upload / AI preference toggles consistent (parent → child).

- When audio context is off, all audio sub-services and speech-to-text are off.
- When auto-thumbnails is off, styled + Thumbnail Studio toggles are off.
- When Thumbnail Studio master is off, engine + persona are off.
- When Drive analysis (trill) is off, telemetry insights service is off.
- When caption writer is off, frame/scene/video analyzer services are off (they feed captions).

IMPORTANT — cascade rules:
The cascade only fires when a parent toggle is **explicitly disabled** by the user
(value is present and falsy). When the parent value is *absent* (None / unset), we
leave the children alone so that runtime defaults can take effect. This matters for
the Thumbnail Studio + Pikzels engine, which default to "on" whenever the server
has ``PIKZELS_API_KEY`` configured — forcing the keys to ``False`` here would mean
the env-driven default is never reached and Pikzels would silently never run.

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


def _explicitly_disabled(d: Mapping[str, Any], snake: str, camel: str) -> bool:
    """True only when the user has set the toggle to a falsy value.

    Returns False when the key is missing entirely or set to ``None`` so that
    the worker's runtime defaults (e.g. Pikzels-on-when-API-key-is-set) survive
    the dependency-normalization pass.
    """
    for k in (snake, camel):
        if k in d and d[k] is not None:
            return not _b(d[k], True)
    return False


def normalize_preferences_dict(d: Dict[str, Any]) -> None:
    """
    Mutate a preferences dict in place. Supports snake_case, camelCase, or both.
    """
    # Third-party voice-emotion (Hume) integration removed — preference stays off.
    _set_pair(d, "ai_service_emotion_signals", "aiServiceEmotionSignals", False)

    if _explicitly_disabled(d, "use_audio_context", "useAudioContext"):
        _set_pair(d, "use_audio_context", "useAudioContext", False)
        _set_pair(d, "audio_transcription", "audioTranscription", False)
        _set_pair(d, "ai_service_speech_to_text", "aiServiceSpeechToText", False)
        _set_pair(d, "ai_service_audio_signals", "aiServiceAudioSignals", False)
        _set_pair(d, "ai_service_music_detection", "aiServiceMusicDetection", False)
        _set_pair(d, "ai_service_audio_summary", "aiServiceAudioSummary", False)
        _set_pair(d, "ai_service_emotion_signals", "aiServiceEmotionSignals", False)

    if _explicitly_disabled(d, "audio_transcription", "audioTranscription"):
        _set_pair(d, "ai_service_speech_to_text", "aiServiceSpeechToText", False)

    # Thumbnail master switch. Only cascade-disable when the user has *explicitly*
    # turned auto-thumbnails off — leaving the value unset means "use the runtime
    # default" (the worker enables the studio pipeline whenever PIKZELS_API_KEY
    # is configured). Forcing ``False`` here was the cause of Pikzels never being
    # invoked even though the integration was wired up.
    if _explicitly_disabled(d, "auto_thumbnails", "autoThumbnails"):
        _set_pair(d, "styled_thumbnails", "styledThumbnails", False)
        _set_pair(d, "thumbnail_studio_enabled", "thumbnailStudioEnabled", False)
        _set_pair(d, "thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled", False)
        _set_pair(d, "thumbnail_persona_enabled", "thumbnailPersonaEnabled", False)

    if _explicitly_disabled(d, "thumbnail_studio_enabled", "thumbnailStudioEnabled"):
        _set_pair(d, "thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled", False)
        _set_pair(d, "thumbnail_persona_enabled", "thumbnailPersonaEnabled", False)

    if _explicitly_disabled(d, "trill_enabled", "trillEnabled"):
        _set_pair(d, "ai_service_telemetry", "aiServiceTelemetry", False)
        _set_pair(d, "trill_ai_enhance", "trillAiEnhance", False)

    if _explicitly_disabled(d, "ai_service_caption_writer", "aiServiceCaptionWriter"):
        _set_pair(d, "ai_service_frame_inspector", "aiServiceFrameInspector", False)
        _set_pair(d, "ai_service_scene_understanding", "aiServiceSceneUnderstanding", False)
        _set_pair(d, "ai_service_video_analyzer", "aiServiceVideoAnalyzer", False)


def normalize_upload_preferences_snake(p: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy normalized; snake_case keys (for user_preferences row)."""
    d = dict(p)
    normalize_preferences_dict(d)
    return d
