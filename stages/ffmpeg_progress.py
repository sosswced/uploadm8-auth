"""Robust FFmpeg stderr progress parsing.

FFmpeg often overwrites progress with ``\\r`` (no newline). asyncio
``StreamReader.readline()`` waits for ``\\n`` and raises LimitOverrunError
(``Separator is not found, and chunk exceed the limit``) once the buffer
hits ~64KB — which kills mid-encode UI heartbeats and makes the queue look
frozen on \"Building platform formats\".
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Optional

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

# Cap in-memory stderr retention during long encodes (keep the tail — FFmpeg
# errors matter at the end). ~4000 chunks × ~4KB ≈ 16MB worst case before trim.
STDERR_KEEP_CHUNKS = 2000


def trim_stderr_buffer(chunks: list, *, keep: int = STDERR_KEEP_CHUNKS) -> None:
    """Drop the oldest half of an FFmpeg stderr buffer once it exceeds 2×keep."""
    if len(chunks) > keep * 2:
        del chunks[: len(chunks) - keep]


async def communicate_or_kill(proc):
    """``proc.communicate()`` that kills the child if the awaiting task is cancelled.

    Stage ``wait_for`` budgets cancel the coroutine, not the subprocess —
    every helper ffmpeg/ffprobe call must go through this (or an equivalent
    handler) so timeouts never orphan children.
    """
    try:
        return await proc.communicate()
    except asyncio.CancelledError:
        await kill_process_quietly(proc)
        raise


async def kill_process_quietly(proc, *, wait_timeout: float = 10.0) -> None:
    """Kill a subprocess left running when its awaiting task was cancelled.

    ``asyncio.wait_for`` stage budgets cancel the *coroutine*, not the child
    process — without this, an orphan FFmpeg keeps burning CPU and holding
    the worker slot. Safe to call on already-exited processes.
    """
    if proc is None:
        return
    try:
        if proc.returncode is not None:
            return
        proc.kill()
    except (ProcessLookupError, OSError):
        return
    except Exception:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=wait_timeout)
    except Exception:
        pass


def parse_ffmpeg_time_seconds(text: str) -> Optional[float]:
    """Extract encoded media time from an FFmpeg progress fragment."""
    if not text:
        return None
    m = _TIME_RE.search(text)
    if not m:
        return None
    try:
        h, mn, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return max(0.0, h * 3600.0 + mn * 60.0 + s)
    except (TypeError, ValueError):
        return None


async def iter_ffmpeg_stderr_text(
    stderr,
    *,
    chunk_size: int = 4096,
) -> AsyncIterator[str]:
    """Yield stderr text fragments split on ``\\r`` / ``\\n`` without LimitOverrunError."""
    if stderr is None:
        return
    buf = b""
    while True:
        try:
            data = await stderr.read(chunk_size)
        except Exception:
            break
        if not data:
            break
        buf += data
        while True:
            n_pos = buf.find(b"\n")
            r_pos = buf.find(b"\r")
            if n_pos < 0 and r_pos < 0:
                break
            if n_pos < 0:
                cut = r_pos
            elif r_pos < 0:
                cut = n_pos
            else:
                cut = min(n_pos, r_pos)
            piece = buf[:cut]
            buf = buf[cut + 1 :]
            if piece:
                yield piece.decode("utf-8", errors="replace")
    if buf:
        yield buf.decode("utf-8", errors="replace")
