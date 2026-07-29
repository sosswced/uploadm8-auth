"""Worker stall hardening: FFmpeg kill-on-cancel, multimodal soft-fail,
dashcam OSD budget, bounded stderr buffers, reclaim task ownership."""

from __future__ import annotations

import asyncio
import inspect
import sys

import pytest

from stages.ffmpeg_progress import (
    STDERR_KEEP_CHUNKS,
    kill_process_quietly,
    trim_stderr_buffer,
)


# ── kill_process_quietly ────────────────────────────────────────────────


def test_kill_process_quietly_terminates_running_process():
    async def _run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.returncode is None
        await kill_process_quietly(proc)
        return proc.returncode

    assert asyncio.run(_run()) is not None


def test_kill_process_quietly_noop_on_exited_and_none():
    async def _run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        rc = proc.returncode
        await kill_process_quietly(proc)  # must not raise on exited process
        assert proc.returncode == rc
        await kill_process_quietly(None)  # must not raise on None
        return True

    assert asyncio.run(_run()) is True


def test_cancelled_await_kills_subprocess():
    """The stage pattern: wait_for cancels the coroutine → child must die."""

    async def _run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def _stage():
            try:
                await proc.communicate()
            except asyncio.CancelledError:
                await kill_process_quietly(proc)
                raise

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_stage(), timeout=0.3)
        # Give the loop a beat to reap.
        await asyncio.sleep(0.1)
        return proc.returncode

    assert asyncio.run(_run()) is not None


def test_worker_ffmpeg_heartbeat_kills_in_finally():
    import worker

    src = inspect.getsource(worker._run_deduplicated_transcode)
    assert "kill_process_quietly(proc)" in src
    # stdout must not be a PIPE that can fill and deadlock the encode.
    assert "stdout=asyncio.subprocess.DEVNULL" in src


def test_stage_subprocess_sites_kill_on_cancel():
    """Every stage that awaits an FFmpeg/ffprobe child must kill it on cancel."""
    import stages.audio_stage as aud
    import stages.caption_stage as cap
    import stages.dashcam_osd_stage as osd
    import stages.tiktok_cover_burn as burn
    import stages.transcode_stage as tc
    import stages.video_intelligence_stage as vi
    import stages.vision_stage as vis
    import stages.watermark_stage as wm

    for mod in (wm, osd, vi, tc, aud):
        src = inspect.getsource(mod)
        assert "kill_process_quietly" in src, mod.__name__
        assert "except asyncio.CancelledError" in src, mod.__name__
    # Helper-based sites route communicate() through communicate_or_kill.
    for mod in (vis, cap, burn):
        src = inspect.getsource(mod)
        assert "communicate_or_kill" in src, mod.__name__


def test_communicate_or_kill_terminates_on_cancel():
    """communicate_or_kill must reap the child when the awaiting task is cancelled."""
    from stages.ffmpeg_progress import communicate_or_kill

    async def _run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(communicate_or_kill(proc), timeout=0.3)
        await asyncio.sleep(0.1)
        return proc.returncode

    assert asyncio.run(_run()) is not None


# ── multimodal soft-fail + gather guard ─────────────────────────────────


def test_multimodal_runners_catch_timeout():
    """Audio/vision/VI/TL runners must soft-catch their wait_for timeouts so
    one slow provider cannot fail the upload."""
    import worker

    src = inspect.getsource(worker.run_processing_pipeline)
    for budget_name in (
        "STAGE_TIMEOUT_AUDIO",
        "STAGE_TIMEOUT_VISION",
        "STAGE_TIMEOUT_VI",
        "STAGE_TIMEOUT_TWELVELABS",
    ):
        assert f"timed out after {{{budget_name}}}" in src, budget_name


def test_multimodal_gather_uses_return_exceptions():
    """The gather must not let one provider's crash cancel siblings/abort."""
    import worker

    src = inspect.getsource(worker.run_processing_pipeline)
    assert "asyncio.gather(*_mm_tasks, return_exceptions=True)" in src
    assert "Multimodal provider soft-failed" in src


def test_gather_return_exceptions_pattern_soft_fails():
    """Behavioral check of the guard pattern: a raising provider does not
    cancel its sibling and the exception is surfaced, not raised."""
    ran = {"ok": False}

    async def _ok():
        await asyncio.sleep(0.05)
        ran["ok"] = True

    async def _boom():
        raise asyncio.TimeoutError("provider budget")

    async def _run():
        return await asyncio.gather(_ok(), _boom(), return_exceptions=True)

    results = asyncio.run(_run())
    assert ran["ok"] is True
    assert any(isinstance(r, asyncio.TimeoutError) for r in results)


# ── dashcam OSD budget ──────────────────────────────────────────────────


def test_dashcam_osd_budget_default_and_env(monkeypatch):
    from stages import pipeline_stage_budgets as budgets

    assert budgets.stage_timeout_dashcam_osd() == 600.0
    assert budgets.get_all_budgets()["dashcam_osd"] == 600.0
    monkeypatch.setenv("STAGE_TIMEOUT_DASHCAM_OSD_SEC", "120")
    assert budgets.stage_timeout_dashcam_osd() == 120.0


def test_worker_wraps_dashcam_osd_in_wait_for():
    import worker

    src = inspect.getsource(worker.run_processing_pipeline)
    assert "run_dashcam_osd_stage(ctx)" in src
    assert "STAGE_TIMEOUT_DASHCAM_OSD" in src
    assert "Dashcam OSD timed out" in src


# ── bounded stderr buffer ───────────────────────────────────────────────


def test_trim_stderr_buffer_keeps_tail():
    chunks = [b"chunk-%d" % i for i in range(STDERR_KEEP_CHUNKS * 2 + 50)]
    trim_stderr_buffer(chunks)
    assert len(chunks) == STDERR_KEEP_CHUNKS
    # Tail preserved — FFmpeg errors show up at the end of stderr.
    assert chunks[-1] == b"chunk-%d" % (STDERR_KEEP_CHUNKS * 2 + 49)


def test_trim_stderr_buffer_noop_under_cap():
    chunks = [b"a"] * 10
    trim_stderr_buffer(chunks)
    assert len(chunks) == 10


# ── reclaim task ownership ──────────────────────────────────────────────


def test_reclaim_tasks_are_referenced_and_logged():
    import worker

    src = inspect.getsource(worker.run_stream_reclaim_loop)
    assert "_reclaim_tasks" in src
    assert "add_done_callback(_reclaim_tasks.discard)" in src
    assert "reclaimed job failed" in src
