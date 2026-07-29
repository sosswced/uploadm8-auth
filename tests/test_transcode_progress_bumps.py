"""Multi-platform transcode must keep UI + updated_at alive during FFmpeg.

Without mid-encode updates the UI freezes at 48% for 15–30+ minutes, client
stale-handoff/TIMEOUT fires, and recovery may treat a healthy encode as orphan.
"""

from __future__ import annotations

import inspect

import worker
from stages.pipeline_stage_budgets import stage_timeout_transcode


def test_deduplicated_transcode_bumps_progress_per_group():
    src = inspect.getsource(worker._run_deduplicated_transcode)
    assert "update_stage_progress" in src
    assert "_bump_transcode_progress" in src
    assert "_wait_ffmpeg_with_heartbeat" in src
    assert "done_groups" in src
    # Fallback ffprobe-fail path must still pass db_pool for mid-encode bumps.
    assert "run_transcode_stage(ctx, db_pool=db_pool)" in src
    # DB blips must not kill encode.
    assert "Transcode progress bump skipped" in src


def test_ffmpeg_wait_uses_heartbeat_and_live_time_parse():
    src = inspect.getsource(worker._run_deduplicated_transcode)
    assert "_wait_ffmpeg_with_heartbeat(proc" in src
    assert "ffmpeg heartbeat" in src
    # Live mid-encode % via chunked stderr parse (not bare communicate() / readline).
    assert "parse_ffmpeg_time_seconds" in src
    assert "iter_ffmpeg_stderr_text" in src
    assert ".readline" not in src
    assert "proc.communicate()" not in src


def test_transcode_progress_band_is_wide_and_stays_below_audio():
    """Bumps must never reach STAGE_PROGRESS['audio']; band must be UI-visible."""
    base = int(worker.STAGE_PROGRESS["transcode"])
    nxt = int(worker.STAGE_PROGRESS["audio"])
    assert base < nxt
    span = max(1, nxt - base - 1)
    assert span >= 10, f"transcode band too narrow ({span}); UI looks frozen"
    for total in (1, 2, 3, 4):
        for done in range(1, total + 1):
            pct = base + max(1, int(span * done / total))
            pct = min(nxt - 1, pct)
            assert base <= pct < nxt
        floor = base
        for tick in range(1, 20):
            micro = min(nxt - 1, floor + min(max(0, span - 1), tick))
            assert micro < nxt


def test_transcode_stage_timeout_default_covers_multi_platform(monkeypatch):
    """Default wall clock must exceed a single 30m encode when 4 groups run."""
    monkeypatch.delenv("STAGE_TIMEOUT_TRANSCODE_SEC", raising=False)
    monkeypatch.delenv("STAGE_TIMEOUT_TRANSCODE_SECONDS", raising=False)
    assert stage_timeout_transcode() >= 3600.0


def test_legacy_transcode_video_uses_stage_progress_not_dead_api():
    from stages import transcode_stage

    src = inspect.getsource(transcode_stage.transcode_video)
    assert "update_stage_progress" in src
    assert "update_upload_progress" not in src
    assert "iter_ffmpeg_stderr_text" in src
    assert ".readline" not in src
