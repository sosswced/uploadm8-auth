"""Random matching .mp4 + .map pairs from a local media library folder.

Used by TUP / live-demo / overnight upload fixtures so each run exercises a
fresh >3‑minute PNW clip (YouTube copyright Shorts trim path).
"""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path
from typing import Optional

# Session cache so video + map resolvers return the same pair.
_LOCK = threading.Lock()
_CACHED_PAIR: Optional[tuple[Path, Path]] = None
_CACHED_LIBRARY: Optional[str] = None

DEFAULT_MEDIA_LIBRARY = Path(r"G:\My Drive\pnw 256\F\F\Normal")

_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
_MAP_EXTS = {".map"}


def e2e_media_library() -> Path | None:
    raw = (os.environ.get("E2E_MEDIA_LIBRARY") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_dir() else None
    if DEFAULT_MEDIA_LIBRARY.is_dir():
        return DEFAULT_MEDIA_LIBRARY
    return None


def _stem_key(path: Path) -> str:
    return path.stem.strip().lower()


def list_matching_pairs(library: Path) -> list[tuple[Path, Path]]:
    """Return (video, map) pairs that share the same basename (case-insensitive)."""
    videos: dict[str, Path] = {}
    maps: dict[str, Path] = {}
    try:
        for entry in library.iterdir():
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            key = _stem_key(entry)
            if ext in _VIDEO_EXTS:
                # Prefer .MP4 over duplicates if any.
                videos.setdefault(key, entry)
            elif ext in _MAP_EXTS:
                maps.setdefault(key, entry)
    except OSError:
        return []
    pairs: list[tuple[Path, Path]] = []
    for key, video in videos.items():
        m = maps.get(key)
        if m is not None:
            pairs.append((video, m))
    return pairs


def pick_random_media_pair(
    library: Path | None = None,
    *,
    force_new: bool = False,
) -> tuple[Path, Path] | None:
    """
    Pick one matching video+.map pair.

    Caches per process so ``e2e_test_video()`` and ``e2e_test_telemetry_map()``
    stay aligned. Override with ``E2E_MEDIA_PAIR_SEED`` for reproducible picks.
    Explicit ``E2E_TEST_VIDEO`` / ``E2E_TEST_TELEMETRY_MAP`` still win in config.
    """
    global _CACHED_PAIR, _CACHED_LIBRARY

    lib = library or e2e_media_library()
    if lib is None:
        return None
    lib_key = str(lib.resolve()) if lib.exists() else str(lib)

    with _LOCK:
        if (
            not force_new
            and _CACHED_PAIR is not None
            and _CACHED_LIBRARY == lib_key
            and _CACHED_PAIR[0].is_file()
            and _CACHED_PAIR[1].is_file()
        ):
            return _CACHED_PAIR

        pairs = list_matching_pairs(lib)
        if not pairs:
            return None

        seed_raw = (os.environ.get("E2E_MEDIA_PAIR_SEED") or "").strip()
        rng = random.Random(seed_raw) if seed_raw else random.Random()
        video, tmap = rng.choice(pairs)
        _CACHED_PAIR = (video, tmap)
        _CACHED_LIBRARY = lib_key
        return _CACHED_PAIR


def clear_media_pair_cache() -> None:
    global _CACHED_PAIR, _CACHED_LIBRARY
    with _LOCK:
        _CACHED_PAIR = None
        _CACHED_LIBRARY = None


def describe_cached_pair() -> dict:
    with _LOCK:
        if _CACHED_PAIR is None:
            return {"selected": False}
        v, m = _CACHED_PAIR
        return {
            "selected": True,
            "library": _CACHED_LIBRARY,
            "video": str(v),
            "map": str(m),
            "stem": v.stem,
        }
