"""Unit tests for transcode HD-tier and encode-parameter helpers."""

from stages.transcode_stage import (
    analyze_transcode_needs,
    build_ffmpeg_command,
    get_platform_spec,
    resolve_x264_encode_params,
    source_hd_tier,
    VideoInfo,
)


def _info(**kwargs) -> VideoInfo:
    defaults = {
        "width": 1920,
        "height": 1080,
        "duration": 60.0,
        "fps": 30.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "video_bitrate": None,
        "audio_bitrate": None,
        "sample_rate": 48000,
        "pixel_format": "yuv420p",
        "rotation": 0,
        "file_size": 0,
    }
    defaults.update(kwargs)
    return VideoInfo(**defaults)


def test_source_hd_tier_1080p_and_4k():
    assert source_hd_tier(_info(width=1920, height=1080)) == "1080p"
    assert source_hd_tier(_info(width=3840, height=2160)) == "4k"
    assert source_hd_tier(_info(width=1280, height=720)) == "standard"


def test_get_platform_spec_boosts_1080p_bitrate():
    spec = get_platform_spec("youtube", _info(width=1920, height=1080))
    assert spec["max_bitrate_video"] == "18M"


def test_resolve_x264_encode_params_1080p_dashcam_defaults():
    preset, crf, tune = resolve_x264_encode_params(_info(width=1920, height=1080))
    assert preset == "fast"
    assert crf == "19"
    assert tune == "film"


def test_analyze_transcode_needs_splits_video_and_audio():
    info = _info(audio_codec="mp3", sample_rate=44100)
    v_r, a_r = analyze_transcode_needs(info, "youtube", "none")
    assert not v_r
    assert any("audio codec" in r for r in a_r)


def test_build_ffmpeg_command_video_copy_when_only_audio_needs_work():
    info = _info(audio_codec="mp3", sample_rate=48000)
    cmd = build_ffmpeg_command(
        __import__("pathlib").Path("in.mp4"),
        __import__("pathlib").Path("out.mp4"),
        info,
        "youtube",
        "none",
    )
    assert "-c:v" in cmd and "copy" in cmd
    assert "-c:a" in cmd and "aac" in cmd


def test_build_ffmpeg_command_trim_only_stream_copy():
    info = _info(duration=200.0)
    cmd = build_ffmpeg_command(
        __import__("pathlib").Path("in.mp4"),
        __import__("pathlib").Path("out.mp4"),
        info,
        "youtube",
        "none",
    )
    assert "-c:v" in cmd and "copy" in cmd
    assert "-c:a" in cmd and "copy" in cmd
    assert "-t" in cmd
