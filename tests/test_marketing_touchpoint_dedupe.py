"""Unit tests for marketing touchpoint dedupe helpers (no DB)."""

from stages.transcode_stage import VideoInfo, get_platform_spec, source_hd_tier


def _dashcam_info() -> VideoInfo:
    return VideoInfo(
        width=1920,
        height=1080,
        duration=120.0,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        video_bitrate=None,
        audio_bitrate=None,
        sample_rate=48000,
        pixel_format="yuv420p",
        rotation=0,
        file_size=0,
    )


def test_hd_tier_boosts_youtube_bitrate_for_dedup_fingerprint():
    info = _dashcam_info()
    assert source_hd_tier(info) == "1080p"
    yt = get_platform_spec("youtube", info)
    tk = get_platform_spec("tiktok", info)
    assert yt["max_bitrate_video"] == "18M"
    assert tk["max_bitrate_video"] == "16M"


def test_platform_spec_fingerprint_includes_bitrate():
    from worker import _platform_spec_fingerprint

    info = _dashcam_info()
    fp_yt = _platform_spec_fingerprint("youtube", info)
    fp_ig = _platform_spec_fingerprint("instagram", info)
    assert "18M" in fp_yt
    assert "14M" in fp_ig
    assert fp_yt != fp_ig
