"""Video Intelligence inline byte cap — Starter-safe clamp."""

from __future__ import annotations


def test_vi_max_bytes_clamps_100mib_on_starter(monkeypatch):
    from stages import video_intelligence_stage as vis

    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    monkeypatch.delenv("VIDEO_INTELLIGENCE_MAX_BYTES_FORCE", raising=False)
    assert vis._parse_vi_max_bytes() == vis._VI_SMALL_PLAN_MAX_BYTES


def test_vi_max_bytes_force_allows_100mib_on_starter(monkeypatch):
    from stages import video_intelligence_stage as vis

    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES_FORCE", "1")
    assert vis._parse_vi_max_bytes() == 100 * 1024 * 1024


def test_vi_max_bytes_100mib_ok_on_standard(monkeypatch):
    from stages import video_intelligence_stage as vis

    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "2048")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    monkeypatch.delenv("VIDEO_INTELLIGENCE_MAX_BYTES_FORCE", raising=False)
    assert vis._parse_vi_max_bytes() == 100 * 1024 * 1024


def test_vi_max_bytes_default_small_plan(monkeypatch):
    from stages import video_intelligence_stage as vis

    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.delenv("VIDEO_INTELLIGENCE_MAX_BYTES", raising=False)
    assert vis._parse_vi_max_bytes() == vis._VI_SMALL_PLAN_MAX_BYTES


def test_vi_stage_enabled_always_true_even_if_env_false(monkeypatch):
    """Infra must not kill VI; Settings → Video Analyzer is the off switch."""
    monkeypatch.setenv("VIDEO_INTELLIGENCE_STAGE_ENABLED", "false")
    import importlib
    import stages.video_intelligence_stage as vis

    importlib.reload(vis)
    assert vis.VIDEO_INTELLIGENCE_STAGE_ENABLED is True
    # Restore default for other tests that import the module later.
    monkeypatch.delenv("VIDEO_INTELLIGENCE_STAGE_ENABLED", raising=False)
    importlib.reload(vis)