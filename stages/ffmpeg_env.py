"""Resolve ffmpeg / ffprobe executables for worker video stages."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def resolve_ffmpeg_executable(binary: str = "ffmpeg") -> Optional[str]:
    """Return path to *binary* (``"ffmpeg"`` or ``"ffprobe"``), or ``None``.

    Environment variable lookup order:
    - ``FFMPEG_BIN``  — path or command name for ``ffmpeg``
    - ``FFPROBE_BIN`` — path or command name for ``ffprobe`` (falls back to
      sibling of resolved ffmpeg when not set)

    Setting these is recommended on Windows dev machines where the tools are
    not on PATH, or in container environments with non-standard install paths.
    """
    env_key = "FFPROBE_BIN" if binary == "ffprobe" else "FFMPEG_BIN"
    raw = (os.environ.get(env_key) or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p)
        w = shutil.which(raw)
        if w:
            return w

    # For ffprobe try to resolve sibling of the ffmpeg binary so they always
    # share the same installation directory.
    if binary == "ffprobe":
        ffmpeg_path = resolve_ffmpeg_executable("ffmpeg")
        if ffmpeg_path:
            sibling = Path(ffmpeg_path).parent / "ffprobe"
            if sibling.is_file():
                return str(sibling)
            sibling_exe = Path(ffmpeg_path).parent / "ffprobe.exe"
            if sibling_exe.is_file():
                return str(sibling_exe)

    return shutil.which(binary)
